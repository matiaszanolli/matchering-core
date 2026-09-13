/**
 * isMacPlatform Regression Tests (#4556)
 *
 * The deprecated `navigator.platform` check that decides ⌘ vs Ctrl used to be
 * duplicated byte-for-byte across `keyboardShortcutsService`,
 * `useKeyboardShortcuts` (twice, counting the `getShortcutString` alias) and
 * `useShortcutFormatting`. A fix applied to one site would have left the others
 * disagreeing, producing inconsistent glyphs within a single session.
 *
 * These tests pin the single helper's fallback chain and assert the
 * duplication does not come back.
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { isMacPlatform } from '../keyboardShortcutsService';

type NavigatorOverrides = {
  userAgentData?: { platform?: string };
  platform?: string;
  userAgent?: string;
};

/** Redefine the navigator properties under test, restoring them afterwards. */
function withNavigator(overrides: NavigatorOverrides, run: () => void): void {
  const keys = ['userAgentData', 'platform', 'userAgent'] as const;
  const saved = keys.map((k) => [k, Object.getOwnPropertyDescriptor(navigator, k)] as const);

  try {
    for (const key of keys) {
      Object.defineProperty(navigator, key, {
        value: overrides[key],
        configurable: true,
        writable: true,
      });
    }
    run();
  } finally {
    for (const [key, descriptor] of saved) {
      if (descriptor) {
        Object.defineProperty(navigator, key, descriptor);
      } else {
        delete (navigator as unknown as Record<string, unknown>)[key];
      }
    }
  }
}

describe('isMacPlatform', () => {
  it('prefers the modern userAgentData.platform hint', () => {
    withNavigator(
      // navigator.platform deliberately disagrees — the modern hint must win.
      { userAgentData: { platform: 'macOS' }, platform: 'Win32', userAgent: 'Windows' },
      () => expect(isMacPlatform()).toBe(true)
    );

    withNavigator(
      { userAgentData: { platform: 'Windows' }, platform: 'MacIntel', userAgent: 'Macintosh' },
      () => expect(isMacPlatform()).toBe(false)
    );
  });

  it('falls back to navigator.platform when userAgentData is absent', () => {
    withNavigator({ platform: 'MacIntel', userAgent: 'Windows' }, () =>
      expect(isMacPlatform()).toBe(true)
    );
    withNavigator({ platform: 'Win32', userAgent: 'Windows' }, () =>
      expect(isMacPlatform()).toBe(false)
    );
  });

  it('falls back to the user-agent string when platform is reduced away', () => {
    withNavigator({ platform: '', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' }, () =>
      expect(isMacPlatform()).toBe(true)
    );
    withNavigator({ platform: '', userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }, () =>
      expect(isMacPlatform()).toBe(false)
    );
  });

  it('is the only place that reads the deprecated navigator.platform', () => {
    const srcRoot = resolve(__dirname, '../..');
    const offenders: string[] = [];

    const walk = (dir: string): void => {
      for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
          if (entry === '__tests__' || entry === 'node_modules' || entry === 'test') continue;
          walk(full);
        } else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
          if (readFileSync(full, 'utf8').includes('navigator.platform')) {
            offenders.push(full.slice(srcRoot.length + 1));
          }
        }
      }
    };
    walk(srcRoot);

    expect(offenders).toEqual(['services/keyboardShortcutsService.ts']);
  });
});
