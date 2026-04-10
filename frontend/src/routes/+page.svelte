<script lang="ts">
	// ── Types ────────────────────────────────────────────────────────────────
	interface CompressResult {
		reconstructed_b64: string;
		psnr: number;
		ssim: number;
		original_size_bytes: number;
		compressed_size_bytes: number;
		compression_ratio: string;
		latent_dim: number;
	}

	// ── State (Svelte 5 runes) ────────────────────────────────────────────────
	let selectedFile: File | null = $state(null);
	let previewUrl: string = $state('');
	let latentDim: number = $state(128);
	let isLoading: boolean = $state(false);
	let errorMessage: string = $state('');
	let result: CompressResult | null = $state(null);
	let isDragging: boolean = $state(false);

	// ── Derived ──────────────────────────────────────────────────────────────
	let hasResults = $derived(result !== null);
	let reconstructedSrc = $derived(
		result ? `data:image/png;base64,${result.reconstructed_b64}` : ''
	);
	let latentSizeLabel = $derived(`${latentDim} floats = ${latentDim * 4} bytes`);

	// ── Auto re-compress when latent dim changes ──────────────────────────────
	$effect(() => {
		// Capture latentDim to establish reactivity dependency
		const dim = latentDim;
		if (selectedFile && !isLoading) {
			handleCompress(dim);
		}
	});

	// ── Handlers ─────────────────────────────────────────────────────────────
	function handleFileSelect(files: FileList | null): void {
		if (!files || files.length === 0) return;
		const file = files[0];
		if (!file.type.startsWith('image/')) {
			errorMessage = 'Please select a PNG or JPG image file.';
			return;
		}
		errorMessage = '';
		result = null;
		selectedFile = file;
		previewUrl = URL.createObjectURL(file);
		handleCompress(latentDim);
	}

	async function handleCompress(dim: number): Promise<void> {
		if (!selectedFile) return;
		isLoading = true;
		errorMessage = '';

		const formData = new FormData();
		formData.append('file', selectedFile);
		formData.append('latent_dim', String(dim));

		try {
			const response = await fetch('/api/compress', {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
				throw new Error(err.detail ?? `Server error ${response.status}`);
			}

			result = (await response.json()) as CompressResult;
		} catch (err) {
			errorMessage =
				err instanceof Error ? err.message : 'Failed to reach the server. Is the backend running?';
			result = null;
		} finally {
			isLoading = false;
		}
	}

	function handleReset(): void {
		if (previewUrl) URL.revokeObjectURL(previewUrl);
		selectedFile = null;
		previewUrl = '';
		result = null;
		errorMessage = '';
		isLoading = false;
	}

	function onInputChange(event: Event): void {
		const input = event.currentTarget as HTMLInputElement;
		handleFileSelect(input.files);
	}

	function onDrop(event: DragEvent): void {
		event.preventDefault();
		isDragging = false;
		handleFileSelect(event.dataTransfer?.files ?? null);
	}

	function onDragOver(event: DragEvent): void {
		event.preventDefault();
		isDragging = true;
	}

	function onDragLeave(): void {
		isDragging = false;
	}
</script>

<svelte:head>
	<title>VAE Medical Image Compression</title>
</svelte:head>

<div class="min-h-screen bg-gray-950 text-gray-100 font-mono">

	<!-- ── Header ──────────────────────────────────────────────────────────── -->
	<header class="border-b border-gray-800 px-6 py-5">
		<div class="max-w-6xl mx-auto">
			<h1 class="text-2xl font-bold text-cyan-400 tracking-tight">
				VAE Medical Image Compression
			</h1>
			<p class="mt-1 text-sm text-gray-400">
				Convolutional Variational Autoencoder · NIH ChestX-ray14 · 256×256 grayscale
			</p>
		</div>
	</header>

	<main class="max-w-6xl mx-auto px-6 py-8 space-y-8">

		<!-- ── Controls panel ──────────────────────────────────────────────── -->
		<section class="grid grid-cols-1 md:grid-cols-2 gap-6">

			<!-- Upload zone -->
			<div>
				<p class="block text-xs uppercase tracking-widest text-gray-400 mb-2">
					Input Image
				</p>
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
						{isDragging
							? 'border-cyan-400 bg-cyan-950/30'
							: selectedFile
								? 'border-gray-600 bg-gray-900'
								: 'border-gray-700 bg-gray-900 hover:border-gray-500'}"
					ondrop={onDrop}
					ondragover={onDragOver}
					ondragleave={onDragLeave}
					role="button"
					tabindex="0"
					onkeydown={(e) => e.key === 'Enter' && (document.getElementById('file-input') as HTMLInputElement)?.click()}
				>
					<input
						id="file-input"
						type="file"
						accept="image/png,image/jpeg"
						class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
						onchange={onInputChange}
					/>

					{#if selectedFile}
						<div class="space-y-2">
							<div class="text-cyan-400 text-sm font-semibold truncate">{selectedFile.name}</div>
							<div class="text-gray-500 text-xs">{(selectedFile.size / 1024).toFixed(1)} KB</div>
							<button
								class="text-xs text-gray-500 hover:text-red-400 transition-colors mt-1 z-10 relative"
								onclick={(e) => { e.stopPropagation(); handleReset(); }}
							>
								✕ clear
							</button>
						</div>
					{:else}
						<div class="space-y-2">
							<div class="text-3xl text-gray-600">⬆</div>
							<p class="text-sm text-gray-400">Drop PNG or JPG here</p>
							<p class="text-xs text-gray-600">or click to browse</p>
						</div>
					{/if}
				</div>
			</div>

			<!-- Latent dim + status -->
			<div class="space-y-5">
				<div>
					<p class="block text-xs uppercase tracking-widest text-gray-400 mb-2">
						Latent Dimension
					</p>
					<div class="flex gap-2">
						{#each [64, 128, 256] as dim}
							<button
								class="flex-1 py-2 px-3 rounded-md text-sm font-semibold border transition-all
									{latentDim === dim
										? 'bg-cyan-600 border-cyan-500 text-white'
										: 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500 hover:text-gray-200'}"
								onclick={() => { latentDim = dim; }}
								disabled={isLoading}
							>
								{dim}
							</button>
						{/each}
					</div>
					<p class="mt-2 text-xs text-gray-500">{latentSizeLabel}</p>
				</div>

				<!-- Status / loading indicator -->
				<div class="rounded-md bg-gray-900 border border-gray-800 p-4 text-sm">
					{#if isLoading}
						<div class="flex items-center gap-2 text-cyan-400">
							<span class="inline-block w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></span>
							Compressing with latent_dim={latentDim}…
						</div>
					{:else if selectedFile && hasResults}
						<div class="text-green-400">✓ Compression complete</div>
					{:else if selectedFile}
						<div class="text-gray-500">Select latent dimension to compress</div>
					{:else}
						<div class="text-gray-600">Upload an image to begin</div>
					{/if}
				</div>

				<!-- Error message -->
				{#if errorMessage}
					<div class="rounded-md border border-red-800 bg-red-950/40 p-3 text-sm text-red-400">
						⚠ {errorMessage}
					</div>
				{/if}
			</div>
		</section>

		<!-- ── Results ─────────────────────────────────────────────────────── -->
		{#if hasResults && result}
			<section class="space-y-6">

				<!-- Image comparison -->
				<div>
					<p class="block text-xs uppercase tracking-widest text-gray-400 mb-3">
						Reconstruction (latent_dim={result.latent_dim})
					</p>
					<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
						<!-- Original -->
						<div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
							<div class="px-3 py-2 border-b border-gray-800 text-xs text-gray-500 uppercase tracking-widest">
								Original
							</div>
							{#if previewUrl}
								<img
									src={previewUrl}
									alt="Original X-ray"
									class="w-full aspect-square object-contain bg-black"
								/>
							{/if}
						</div>

						<!-- Reconstructed -->
						<div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
							<div class="px-3 py-2 border-b border-gray-800 text-xs text-gray-500 uppercase tracking-widest">
								Reconstructed
							</div>
							<img
								src={reconstructedSrc}
								alt="Reconstructed X-ray"
								class="w-full aspect-square object-contain bg-black"
							/>
						</div>
					</div>
				</div>

				<!-- Metrics cards -->
				<div>
					<p class="block text-xs uppercase tracking-widest text-gray-400 mb-3">
						Compression Metrics
					</p>
					<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
						<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
							<div class="text-xs text-gray-500 uppercase tracking-wider mb-1">PSNR</div>
							<div class="text-xl font-bold text-cyan-400">{result.psnr}</div>
							<div class="text-xs text-gray-600 mt-1">dB</div>
						</div>

						<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
							<div class="text-xs text-gray-500 uppercase tracking-wider mb-1">SSIM</div>
							<div class="text-xl font-bold text-cyan-400">
								{(result.ssim * 100).toFixed(1)}%
							</div>
							<div class="text-xs text-gray-600 mt-1">structural similarity</div>
						</div>

						<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
							<div class="text-xs text-gray-500 uppercase tracking-wider mb-1">Ratio</div>
							<div class="text-xl font-bold text-cyan-400">{result.compression_ratio}</div>
							<div class="text-xs text-gray-600 mt-1">
								{result.original_size_bytes.toLocaleString()} → {result.compressed_size_bytes} bytes
							</div>
						</div>

						<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
							<div class="text-xs text-gray-500 uppercase tracking-wider mb-1">Latent</div>
							<div class="text-xl font-bold text-cyan-400">{result.latent_dim}</div>
							<div class="text-xs text-gray-600 mt-1">{latentSizeLabel}</div>
						</div>
					</div>
				</div>
			</section>
		{/if}

	</main>
</div>
