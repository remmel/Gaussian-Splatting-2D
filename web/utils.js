export function hexToRGB(hex) {
    // Remove the hash at the start if it's there
    hex = hex.replace(/^#/, '')

    // Parse the hex string
    let bigint = parseInt(hex, 16)

    // Extract R, G, B values
    let r = (bigint >> 16) & 255
    let g = (bigint >> 8) & 255
    let b = bigint & 255

    // Convert to 0-1 range
    return {
        r: r / 255,
        g: g / 255,
        b: b / 255
    }
}
