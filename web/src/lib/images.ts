export const MAX_IMAGE_BYTES = 15 * 1024 * 1024
export const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp']

/** Returns a learner-facing problem with the file, or null when it can be uploaded. */
export function validateImage(file: { type: string; size: number }): string | null {
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) return 'Please choose a PNG, JPEG, or WebP photo.'
  if (file.size > MAX_IMAGE_BYTES) return 'That photo is larger than 15 MB. Try a smaller one.'
  return null
}
