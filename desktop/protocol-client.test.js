'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// #4853: app.setAsDefaultProtocolClient('auralis') was registered with no
// open-url (macOS) or second-instance (Windows/Linux) handler to receive the
// invoked URL -- a live OS-level registration (any local process or a
// crafted link could already invoke auralis://... and have the OS
// launch/foreground the app) with nothing built yet to validate a payload.
// Removed rather than paired with a handler, since no deep-link feature
// exists yet to receive one. This pins that if the registration ever comes
// back, it doesn't come back alone -- a payload must be treated as untrusted
// input from day one, not bolted on after the OS-level surface is already live.
const MAIN_SRC = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

test('setAsDefaultProtocolClient is either absent or paired with a URL-receiving handler', () => {
  if (!MAIN_SRC.includes('setAsDefaultProtocolClient')) {
    return; // nothing registered today -- nothing to pair with a handler
  }
  const hasHandler =
    /app\.on\(\s*'open-url'/.test(MAIN_SRC) || /app\.on\(\s*'second-instance'/.test(MAIN_SRC);
  assert.ok(
    hasHandler,
    "setAsDefaultProtocolClient is registered but neither 'open-url' nor 'second-instance' is handled"
  );
});
