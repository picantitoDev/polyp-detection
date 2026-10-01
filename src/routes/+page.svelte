<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Room,
		createLocalVideoTrack,
		type LocalVideoTrack
	} from 'livekit-client';
	import { env } from '$env/dynamic/public';

	let videoElement: HTMLVideoElement;
	let status = $state('Starting...');

	onMount(async () => {
		try {
			status = 'Getting token...';

			const response = await fetch('/api/livekit/token');

			if (!response.ok) {
				throw new Error('Failed to get LiveKit token');
			}

			const { token } = await response.json();

			status = 'Connecting to LiveKit...';

			const room = new Room();

			await room.connect(env.PUBLIC_LIVEKIT_URL, token);

			status = 'Connected. Starting camera...';

			const cameraTrack: LocalVideoTrack = await createLocalVideoTrack();

			status = 'Camera ready';

			cameraTrack.attach(videoElement);

			await room.localParticipant.publishTrack(cameraTrack);

			status = 'Camera published!';
		} catch (error) {
			console.error(error);
			status = 'Error. Check browser console.';
		}
	});
</script>

<svelte:head>
	<title>Polyp Detection MVP</title>
</svelte:head>

<div class="min-h-screen bg-slate-950 p-8 text-white">
	<div class="mx-auto max-w-4xl">
		<h1 class="mb-2 text-3xl font-bold">Polyp Detection MVP</h1>

		<p class="mb-6 text-slate-400">
			Status: {status}
		</p>

		<div class="overflow-hidden rounded-xl bg-black">
			<video
				bind:this={videoElement}
				autoplay
				playsinline
				muted
				class="aspect-video w-full object-cover"
			></video>
		</div>
	</div>
</div>