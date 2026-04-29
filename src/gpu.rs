use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Cached bind group keyed by a string built from pipeline name + buffer pointer addresses
type BindGroupKey = String;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    bind_group_cache: HashMap<BindGroupKey, wgpu::BindGroup>,
    encoder: Option<wgpu::CommandEncoder>,
    pending_dispatches: u32,
    max_storage_binding: u64,
    pipeline_cache: Option<wgpu::PipelineCache>,
    has_pipeline_cache_feature: bool,
    /// Pre-allocated probe buffers for Poll-mode flush_and_wait (avoids per-call allocation).
    flush_probe_src: Option<wgpu::Buffer>,
    flush_probe_dst: Option<wgpu::Buffer>,
    /// Whether flush_probe_dst is currently in a mapped state.
    flush_probe_mapped: bool,
}

/// Per-request GPU context wrapper.
/// Wraps the stable GpuContext in Arc and adds request-specific state.
pub struct GpuRequestContext {
    pub gpu: std::sync::Arc<GpuContext>,
    /// Active model role for logging ("fast" | "deep" | "asr" | "omni" | "?").
    /// Set by thinker_impl at request dispatch and on slot swap.
    pub role_tag: &'static str,
    /// Optional per-token streaming sink. Set per-request from
    /// `ThinkerRequest.stream_tx`; cleared at the end of the request.
    /// When Some alongside `stream_tokenizer`, the decode loop
    /// (inference.rs) decodes each newly-generated token and pushes
    /// a `StreamChunk { delta_text, finish_reason: None }`. Cleared
    /// after generation so the channel sender drops and SSE writers
    /// can detect end-of-stream.
    pub stream_tx: Option<async_channel::Sender<common::handles::StreamChunk>>,
    /// Tokenizer used to decode tokens for streaming. Wrapped in Arc
    /// so we don't fight the borrow checker when swapping it across
    /// requests on the same long-lived GPU thread. None means "skip
    /// streaming even if `stream_tx` is set".
    pub stream_tokenizer: Option<std::sync::Arc<common::tokenizer::Tokenizer>>,
    /// Whether to disable think-injection for the current request.
    /// Read from `inference_config.disable_think_injection`. Kept as a
    /// bare bool too so the hot path inside `generate_inner` doesn't
    /// have to go through the struct on every iteration. Mirrors what
    /// thinker_impl writes from `ThinkerRequest.disable_think_injection`.
    pub disable_think_injection: bool,
    /// Bag of per-request knobs that aren't worth their own field on
    /// this struct. Set by the thinker dispatch path before each
    /// generation and reset to `Default::default()` afterward — same
    /// lifecycle as `stream_tx` / `stream_tokenizer`.
    pub inference_config: common::handles::InferenceConfig,
}

impl std::ops::Deref for GpuRequestContext {
    type Target = GpuContext;

    fn deref(&self) -> &Self::Target {
        &self.gpu
    }
}

impl std::ops::DerefMut for GpuRequestContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::get_mut(&mut self.gpu).expect("GpuRequestContext wraps shared GpuContext; expected exclusive access")
    }
}

impl GpuRequestContext {
    pub fn new(gpu: GpuContext) -> Self {
        Self {
            gpu: Arc::new(gpu),
            role_tag: "?",
            stream_tx: None,
            stream_tokenizer: None,
            disable_think_injection: false,
            inference_config: common::handles::InferenceConfig::default(),
        }
    }
}

impl GpuContext {
    pub fn new() -> Self {
        pollster::block_on(Self::init())
    }

    /// Create a GpuContext from an externally-owned device and queue.
    /// This allows the caller to create a surface-compatible device and
    /// share it with both the UI renderer and the compute pipeline.
    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let limit = device.limits().max_storage_buffer_binding_size;
        let mut ctx = Self {
            device,
            queue,
            pipelines: HashMap::new(),
            bind_group_cache: HashMap::new(),
            encoder: None,
            pending_dispatches: 0,
            max_storage_binding: limit,
            pipeline_cache: None,
            has_pipeline_cache_feature: false,
            flush_probe_src: None,
            flush_probe_dst: None,
            flush_probe_mapped: false,
        };
        ctx.init_flush_probe();
        ctx
    }

    pub fn max_storage_binding_size(&self) -> u64 {
        self.max_storage_binding
    }


    async fn init() -> Self {
        log::info!("[shady-thinker] requesting GPU adapter (Vulkan preferred)...");
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no suitable GPU adapter found");

        let info = adapter.get_info();
        log::info!("[shady-thinker] GPU: {} (backend={:?}, type={:?})",
            info.name, info.backend, info.device_type);

        let mut limits = adapter.limits();
        log::info!(
            "Adapter limits: max_buffer={}MB, max_storage_binding={}MB",
            limits.max_buffer_size / (1024 * 1024),
            limits.max_storage_buffer_binding_size / (1024 * 1024),
        );
        // For LLM: ensure large enough for embedding tables (~1GB for 248K vocab).
        // For ASR encoder: adapter defaults are sufficient (~128MB).
        // Only request larger if the adapter already supports it.
        if limits.max_storage_buffer_binding_size >= (1u64 << 30) {
            limits.max_buffer_size = limits.max_buffer_size.max(1u64 << 31);
        }
        // Don't override max_storage_buffer_binding_size — use adapter's native limit.

        // Enable PIPELINE_CACHE if the adapter supports it (speeds up shader compilation on Android).
        let pipeline_cache_feature = if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            log::info!("[shady-thinker] PIPELINE_CACHE feature available");
            wgpu::Features::PIPELINE_CACHE
        } else {
            log::info!("[shady-thinker] PIPELINE_CACHE feature not available");
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("shady-thinker"),
                required_features: pipeline_cache_feature,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("failed to create device");

        let has_pc = pipeline_cache_feature.contains(wgpu::Features::PIPELINE_CACHE);
        let max_storage_binding = device.limits().max_storage_buffer_binding_size;
        let mut ctx = Self {
            device,
            queue,
            pipelines: HashMap::new(),
            bind_group_cache: HashMap::new(),
            encoder: None,
            pending_dispatches: 0,
            max_storage_binding,
            pipeline_cache: None,
            has_pipeline_cache_feature: has_pc,
            flush_probe_src: None,
            flush_probe_dst: None,
            flush_probe_mapped: false,
        };
        ctx.init_flush_probe();
        ctx
    }

    pub fn supports_pipeline_cache(&self) -> bool {
        self.has_pipeline_cache_feature
    }

    fn init_flush_probe(&mut self) {
        let src = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flush_probe_src"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst = self.create_readback_buffer("flush_probe_dst", 4);
        self.flush_probe_src = Some(src);
        self.flush_probe_dst = Some(dst);
    }

    /// Load a Vulkan pipeline cache from raw bytes (previously returned by `get_pipeline_cache_data`).
    /// This dramatically speeds up pipeline compilation on subsequent runs.
    pub fn load_pipeline_cache(&mut self, data: &[u8]) {
        // SAFETY: data was previously returned by get_pipeline_cache_data from the same device family.
        // We use fallback=true so an incompatible cache is silently ignored.
        let cache = unsafe {
            self.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("shady-thinker"),
                data: Some(data),
                fallback: true,
            })
        };
        self.pipeline_cache = Some(cache);
        log::info!("[shady-thinker] pipeline cache loaded ({} bytes)", data.len());
    }

    /// Create an empty pipeline cache (for first-run, enables saving after compilation).
    pub fn create_pipeline_cache(&mut self) {
        let cache = unsafe {
            self.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("shady-thinker"),
                data: None,
                fallback: true,
            })
        };
        self.pipeline_cache = Some(cache);
    }

    /// Get the current pipeline cache data for persistence.
    pub fn get_pipeline_cache_data(&self) -> Option<Vec<u8>> {
        self.pipeline_cache.as_ref().and_then(|c| c.get_data())
    }

    pub fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let aligned = (size + 3) & !3;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn create_storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(
            label,
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub fn upload_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let buffer = self.create_storage_buffer(label, data.len() as u64);
        // Write in chunks to handle large buffers (>1GB)
        let chunk_size = 64 * 1024 * 1024; // 64MB chunks
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let offset = (i * chunk_size) as u64;
            self.write_buffer(&buffer, offset, chunk);
        }
        // Submit staging writes without blocking (no device.poll).
        // Vulkan in-order queue guarantees these copies complete before any
        // subsequent compute dispatches that read from this buffer.
        // This avoids exhausting PowerVR's limited fence/poll resources during model loading.
        self.queue.submit(std::iter::empty());
        buffer
    }

    pub fn create_readback_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(
            label,
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    fn ensure_pipeline(&mut self, name: &str, shader_src: &str) {
        if !self.pipelines.contains_key(name) {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });
            let pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: self.pipeline_cache.as_ref(),
                    });
            self.pipelines.insert(name.to_string(), pipeline);
        }
    }

    fn get_or_create_bind_group(
        &mut self,
        pipeline_name: &str,
        buffers: &[(u32, &wgpu::Buffer)],
    ) -> &wgpu::BindGroup {
        let key = make_bg_key(pipeline_name, buffers);

        if !self.bind_group_cache.contains_key(&key) {
            let pipeline = &self.pipelines[pipeline_name];
            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            self.bind_group_cache.insert(key.clone(), bg);
        }
        &self.bind_group_cache[&key]
    }

    fn ensure_encoder(&mut self) {
        if self.encoder.is_none() {
            self.encoder = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("batch"),
                    }),
            );
        }
    }

    /// Queue a compute dispatch into the current batch encoder.
    pub fn dispatch(
        &mut self,
        pipeline_name: &str,
        shader_src: &str,
        buffers: &[(u32, &wgpu::Buffer)],
        workgroups: (u32, u32, u32),
    ) {
        self.ensure_pipeline(pipeline_name, shader_src);

        // Create bind group (can't borrow self mutably and immutably, so do it in steps)
        let key = make_bg_key(pipeline_name, buffers);

        if !self.bind_group_cache.contains_key(&key) {
            let pipeline = &self.pipelines[pipeline_name];
            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            self.bind_group_cache.insert(key.clone(), bg);
        }

        self.ensure_encoder();
        let encoder = self.encoder.as_mut().unwrap();
        let pipeline = &self.pipelines[pipeline_name];
        let bind_group = &self.bind_group_cache[&key];

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pipeline_name),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.pending_dispatches += 1;
    }

    /// Dispatch with buffer sub-ranges (offset + optional size).
    /// Used for packed MoE expert buffers where each expert is at a different offset.
    /// Bind groups with offsets are NOT cached (each expert dispatch is unique).
    /// Dispatch with buffer sub-ranges (offset + optional size).
    /// Used for packed MoE expert buffers where each expert is at a different offset.
    pub fn dispatch_with_offsets(
        &mut self,
        pipeline_name: &str,
        shader_src: &str,
        buffers: &[(u32, &wgpu::Buffer, u64, Option<u64>)],
        workgroups: (u32, u32, u32),
    ) {
        self.ensure_pipeline(pipeline_name, shader_src);

        // Create bind group with sub-buffer ranges (not cached — each expert is unique)
        let bg = {
            let pipeline = &self.pipelines[pipeline_name];
            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .map(|(binding, buffer, offset, size)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: *offset,
                        size: size.map(|s| std::num::NonZeroU64::new(s).unwrap()),
                    }),
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            })
        };

        self.ensure_encoder();
        let encoder = self.encoder.as_mut().unwrap();
        let pipeline = &self.pipelines[pipeline_name];

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pipeline_name),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.pending_dispatches += 1;
    }

    /// Flush all pending dispatches to the GPU.
    pub fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit(std::iter::once(encoder.finish()));
            self.pending_dispatches = 0;
        }
    }

    /// Flush and block until the GPU has finished all submitted work.
    /// Uses Poll-mode loop with a cached COPY_SRC → MAP_READ probe to avoid
    /// the PowerVR Vulkan driver hang in vkWaitForFences(wait_indefinitely).
    pub fn flush_and_wait(&mut self) {
        self.flush();
        // Submit a copy of the cached probe_src → probe_dst. Vulkan in-order queue
        // guarantees this completes after all prior submitted work. map_async on
        // probe_dst fires only after this fence — i.e., after all prior work is done.
        let probe_src = self.flush_probe_src.as_ref().expect("flush_probe_src not init");
        let probe_dst = self.flush_probe_dst.as_ref().expect("flush_probe_dst not init");

        // Unmap probe_dst if it's still mapped from a previous call
        if self.flush_probe_mapped {
            probe_dst.unmap();
            self.flush_probe_mapped = false;
        }

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flush_probe"),
        });
        enc.copy_buffer_to_buffer(probe_src, 0, probe_dst, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = probe_dst.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.flush_probe_mapped = true;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let _ = self.device.poll(wgpu::PollType::Poll);
            match rx.try_recv() {
                Ok(_) => {
                    probe_dst.unmap();
                    self.flush_probe_mapped = false;
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if std::time::Instant::now() >= deadline {
                        log::warn!("[gpu] flush_and_wait: timed out after 30s — GPU may be hung");
                        return; // leave mapped — next call will unmap
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }
        }
    }

    /// Copy buffer contents (batched into current encoder).
    pub fn copy_buffer(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        self.ensure_encoder();
        let encoder = self.encoder.as_mut().unwrap();
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
    }

    pub fn copy_buffer_offset(&mut self, src: &wgpu::Buffer, src_off: u64,
                               dst: &wgpu::Buffer, dst_off: u64, size: u64) {
        self.ensure_encoder();
        let encoder = self.encoder.as_mut().unwrap();
        encoder.copy_buffer_to_buffer(src, src_off, dst, dst_off, size);
    }

    /// Read back a buffer to CPU. Flushes pending work first.
    /// Uses Poll-mode loop to avoid PowerVR hang in vkWaitForFences.
    pub fn read_buffer(&mut self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        self.try_read_buffer_offset(buffer, 0, size, std::time::Duration::from_secs(30))
            .unwrap_or_else(|| {
                log::warn!("[gpu] read_buffer timed out — returning zeros ({} bytes)", size);
                vec![0u8; size as usize]
            })
    }

    /// Read a sub-range of a buffer to CPU.
    pub fn read_buffer_offset(&mut self, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        self.flush();
        let staging = self.create_readback_buffer("readback_off", size);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_off"),
        });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
        self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    /// Read a sub-range of a buffer with a timeout. Returns None if the GPU
    /// does not respond within `timeout`. Needed on PowerVR where
    /// vkWaitForFences hangs after heavy prefill workloads.
    pub fn try_read_buffer_offset(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
        timeout: std::time::Duration,
    ) -> Option<Vec<u8>> {
        self.flush();
        let staging = self.create_readback_buffer("readback_timed", size);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_timed"),
        });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });

        let deadline = std::time::Instant::now() + timeout;
        loop {
            let _ = self.device.poll(wgpu::PollType::Poll);
            match rx.try_recv() {
                Ok(Ok(())) => {
                    let data = slice.get_mapped_range().to_vec();
                    staging.unmap();
                    return Some(data);
                }
                Ok(Err(e)) => {
                    log::warn!("[gpu] readback mapping failed: {:?}", e);
                    return None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if std::time::Instant::now() >= deadline {
                        log::warn!("[gpu] readback timed out after {:.1}s", timeout.as_secs_f32());
                        return None;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
            }
        }
    }

    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }
}

/// Helper: (binding, buffer) pair
pub fn bind(binding: u32, buffer: &wgpu::Buffer) -> (u32, &wgpu::Buffer) {
    (binding, buffer)
}

/// Build a bind group cache key from pipeline name + buffer pointer addresses.
fn make_bg_key(pipeline_name: &str, buffers: &[(u32, &wgpu::Buffer)]) -> BindGroupKey {
    use std::fmt::Write;
    let mut key = String::with_capacity(pipeline_name.len() + buffers.len() * 20);
    key.push_str(pipeline_name);
    for (binding, buf) in buffers {
        write!(key, ":{binding}:{:p}", &**buf).unwrap();
    }
    key
}
