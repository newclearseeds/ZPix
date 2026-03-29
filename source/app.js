/*
 * ZPix Gradio app custom JavaScript.
 */

// Delegate events for some elements.
document.addEventListener("click", (event) => {
    if (event.target.closest("#quit-app-btn")) {
        return quitApp()
    }

    if (event.target.closest("#import-image-metadata-btn")) {
        return importImageMetadata()
    }

    if (event.target.closest("#swap-lora-btn")) {
        return swapLora()
    }

    /** @type {HTMLAnchorElement | null} */
    const link = event.target.closest("a")

    if (link && link.target === "_blank") {
        openExternalLink(link, event)
    }
})

/**
 * Open a link in default browser.
 *
 * @param {HTMLAnchorElement} link
 * @param {Event} event
 */
function openExternalLink(link, event) {
    event.preventDefault() // Instead of opening a new webview
    // thanks to a custom binding. See webview.cpp
    window.openWithDefaultBrowser(link.href)
}

/**
 * Close the desktop app gracefully.
 */
function quitApp() {
    if (typeof window.quitNativeApp === "function") {
        window.quitNativeApp()
        return
    }

    alert("Close this browser tab to stop viewing the app. The local server may still keep running in the terminal.")
}

/**
 * Import metadata from a generated PNG image.
 */
async function importImageMetadata() {
    /** @type {string} */
    const path = await window.openNativeFileDialog()

    if (!path) return

    /** @type {HTMLTextAreaElement} */
    const portal = document.querySelector("#import-image-path textarea")

    portal.value = `${path}|${Math.floor(Date.now() / 1000)}`
    portal.dispatchEvent(new Event("input"))
}

/**
 * Swap LoRA.
 */
async function swapLora() {
    /**
     * Absolute path to LoRA file selected by user.
     * @type {string}
     */
    const path = await window.openNativeFileDialog()
    // This avoids an unnecessary local upload. See webview.cpp

    if (!path) return // When the user cancels.

    /** @type {HTMLTextAreaElement} */
    const portal = document.querySelector("#lora-path textarea")

    // We place the LoRA path in a "portal" hidden textarea.
    // This "portal" transfers data and control to the backend.
    // Appending a timestamp forces the change event to fire
    // when the user loads-unloads-reloads the same LoRA file.
    portal.value = `${path}|${Math.floor(Date.now() / 1000)}`
    portal.dispatchEvent(new Event("input"))
}
