'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// preload.js requires `window` and the real electron API at module scope, so
// it can't be require()d outside a BrowserWindow. These checks read both
// files as text instead of executing them (#4851).
const PRELOAD_SRC = fs.readFileSync(path.join(__dirname, 'preload.js'), 'utf8');
const MAIN_SRC = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

function invokedChannels(src) {
  const re = /ipcRenderer\.invoke\(\s*'([^']+)'/g;
  const channels = new Set();
  let m;
  while ((m = re.exec(src))) channels.add(m[1]);
  return channels;
}

function handledChannels(src) {
  const re = /ipcMain\.handle\(\s*'([^']+)'/g;
  const channels = new Set();
  let m;
  while ((m = re.exec(src))) channels.add(m[1]);
  return channels;
}

// #4851: preload.js exposed sendToBackend/openExternal invoking
// 'backend-message'/'open-external', neither of which main.js ever handled --
// a no-op today, but a landmine for whoever wires one up later without
// noticing the missing validation main.js's real openExternalSafely()
// provides. Removed rather than implemented (zero renderer consumers of
// either method). This pins the invariant so a future preload addition can't
// silently reopen the same gap.
test('every channel preload.js invokes has a matching ipcMain.handle in main.js', () => {
  const invoked = invokedChannels(PRELOAD_SRC);
  const handled = handledChannels(MAIN_SRC);

  assert.ok(invoked.size > 0, 'sanity check: preload.js should invoke at least one channel');

  const missing = [...invoked].filter((channel) => !handled.has(channel));
  assert.deepEqual(
    missing,
    [],
    `preload.js exposes a channel with no ipcMain.handle: ${missing.join(', ')}`
  );
});

test('the two dead channels do not reappear', () => {
  assert.equal(PRELOAD_SRC.includes('sendToBackend'), false);
  assert.equal(PRELOAD_SRC.includes('backend-message'), false);
  assert.equal(PRELOAD_SRC.includes('openExternal'), false);
  assert.equal(PRELOAD_SRC.includes("'open-external'"), false);
});
