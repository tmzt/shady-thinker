use std::collections::HashMap;
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
}

impl GpuContext {
    pub fn new() -> Self {
        pollster::block_on(Self::init())
    }

    async fn init() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no suitable GPU adapter found");

        log::info!("GPU: {}", adapter.get_info().name);

        let mut limits = adapter.limits();
        log::info!(
            "Adapter limits: max_buffer={}MB, max_storage_binding={}MB",
            limits.max_buffer_size / (1024 * 1024),
            limits.max_storage_buffer_binding_size / (1024 * 1024),
        );
        // Ensure large enough for embedding tables (~1GB for 248K vocab)
        limits.max_buffer_size = limits.max_buffer_size.max(1u64 << 31);
        limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(1u32 << 30);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("shady-thinker"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("failed to create device");

        Self {
            device,
            queue,
            pipelines: HashMap::new(),
            bind_group_cache: HashMap::new(),
            encoder: None,
            pending_dispatches: 0,
        }
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
            self.queue.write_buffer(&buffer, offset, chunk);
        }
        // Flush and wait for completion
        self.queue.submit(std::iter::empty());
        self.device.poll(wgpu::Maintain::Wait);
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
                        cache: None,
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

    /// Flush all pending dispatches to the GPU.
    pub fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit(std::iter::once(encoder.finish()));
            self.pending_dispatches = 0;
        }
    }

    /// Copy buffer contents (batched into current encoder).
    pub fn copy_buffer(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        self.ensure_encoder();
        let encoder = self.encoder.as_mut().unwrap();
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
    }

    /// Read back a buffer to CPU. Flushes pending work first.
    pub fn read_buffer(&mut self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        // Flush any pending dispatches
        self.flush();

        let staging = self.create_readback_buffer("readback", size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
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
