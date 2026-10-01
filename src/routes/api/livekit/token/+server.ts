import { json } from '@sveltejs/kit';
import { AccessToken } from 'livekit-server-sdk';
import { env } from '$env/dynamic/private';

export async function GET() {
	const roomName = 'polyp-mvp';
	const identity = `browser-${crypto.randomUUID()}`;

	const token = new AccessToken(env.LIVEKIT_API_KEY, env.LIVEKIT_API_SECRET, {
		identity,
		ttl: '10m'
	});

	token.addGrant({
		roomJoin: true,
		room: roomName,
		canPublish: true,
		canSubscribe: true,
		canPublishData: true
	});

	return json({
		token: await token.toJwt(),
		roomName
	});
}