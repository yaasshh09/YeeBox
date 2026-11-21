// Scenes in URLs: "#s=1." then base64url of the deflate-raw compressed scene
// JSON. The "1." is the link format version; a future format adds "2." and
// keeps decoding "1.". The JSON itself is versioned by the Rust scene, so
// opening a link reproduces the scene exactly.

export const LINK_PREFIX = "#s=1.";
/** Longer links break in chat apps and some browsers; offer a file instead. */
export const MAX_LINK = 8000;

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Blob([bytes as BlobPart]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(text: string): Uint8Array {
  const b64 = text.replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

/** The URL fragment for a scene. */
export async function encodeScene(json: string): Promise<string> {
  const packed = await pipe(new TextEncoder().encode(json), new CompressionStream("deflate-raw"));
  return LINK_PREFIX + toBase64Url(packed);
}

/** The scene JSON in a fragment, or null when there is none. Throws on a damaged link. */
export async function decodeScene(hash: string): Promise<string | null> {
  if (!hash.startsWith("#s=")) return null;
  if (!hash.startsWith(LINK_PREFIX)) throw new Error("This link was made by a newer version of YeeBox.");
  const bytes = await pipe(fromBase64Url(hash.slice(LINK_PREFIX.length)), new DecompressionStream("deflate-raw"));
  return new TextDecoder().decode(bytes);
}
