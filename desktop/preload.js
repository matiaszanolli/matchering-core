const { contextBridge, ipcRenderer } = require('electron');

// Defense in depth for #4858. This script is attached to the BrowserWindow,
// not to a URL, so it re-runs on every navigation of that window whatever the
// destination. main.js's will-navigate/will-redirect guard is the primary
// control; this second check means that if a navigation ever does slip past
// it, the remote document still gets no IPC surface at all.
//
// The renderer only ever legitimately runs on http://localhost:{8765,3000}.
// The frontend feature-detects via `window.electronAPI !== undefined`
// (utils/electron.ts), so withholding it degrades to plain-web behaviour
// rather than throwing. error.html (file://) does not use the API.
if (window.location.hostname !== 'localhost') {
  console.warn(
    `Auralis preload: refusing to expose electronAPI on non-local origin ${window.location.origin}`
  );
} else {

// Expose safe APIs to renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // File system operations
  selectFile: () => ipcRenderer.invoke('select-file'),
  selectFolder: () => ipcRenderer.invoke('select-folder'),

  // Window control
  minimize: () => ipcRenderer.invoke('window-minimize'),
  maximize: () => ipcRenderer.invoke('window-maximize'),
  close: () => ipcRenderer.invoke('window-close'),

  // Platform info
  platform: process.platform,
  isPackaged: process.env.NODE_ENV === 'production',

  // App info
  version: process.env.npm_package_version || '1.0.0',

  // Event listeners for app state
  onAppReady: (callback) => {
    ipcRenderer.on('app-ready', callback);
  },

  onBackendReady: (callback) => {
    ipcRenderer.on('backend-ready', callback);
  },

  onAppError: (callback) => {
    ipcRenderer.on('app-error', callback);
  },

  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});

}  // end localhost-only exposure guard (#4858)

// Log that preload script has loaded
console.log('Auralis preload script loaded');
console.log('Platform:', process.platform);
console.log('Node version:', process.versions.node);
console.log('Electron version:', process.versions.electron);