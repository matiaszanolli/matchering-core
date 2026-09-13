# MSW Quick Start Guide

## Installation

```bash
cd /mnt/data/src/matchering/auralis-web/frontend
npm install  # MSW v2.6.5 already in package.json
```

## Running Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- --run src/tests/api-integration/library-api.test.ts

# Run tests in watch mode
npm test -- --watch

# Run with coverage
npm run test:coverage
```

## Writing a New Component Integration Test

### 1. Create Test File
```bash
touch src/tests/integration/my-component.test.tsx
```

### 2. Basic Template
```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AllProviders } from '../../test/test-utils';
import MyComponent from '../../components/MyComponent';

describe('MyComponent Integration', () => {
  it('should render with API data', async () => {
    render(
      <AllProviders>
        <MyComponent />
      </AllProviders>
    );

    // Wait for API call
    const element = await screen.findByText('Test Track 1');
    expect(element).toBeInTheDocument();
  });

  it('should handle user interaction', async () => {
    const user = userEvent.setup();

    render(
      <AllProviders>
        <MyComponent />
      </AllProviders>
    );

    const button = await screen.findByRole('button', { name: /play/i });
    await user.click(button);

    // Verify state change
    expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
  });
});
```

## Writing a New API Integration Test

### 1. Create Test File
```bash
touch src/tests/api-integration/my-api.test.ts
```

### 2. Basic Template
```typescript
import { describe, it, expect } from 'vitest';

const API_BASE = 'http://localhost:8765/api';

describe('My API Integration', () => {
  it('should fetch data', async () => {
    const response = await fetch(`${API_BASE}/my-endpoint`);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toHaveProperty('expectedField');
  });

  it('should handle errors', async () => {
    const response = await fetch(`${API_BASE}/invalid-endpoint`);
    expect(response.status).toBe(404);
  });
});
```

## Using Test Helpers

```typescript
import {
  waitForElement,
  waitForApiCall,
  waitForLoadingToFinish,
  typeWithDelay
} from '../../test/utils/test-helpers';

// Wait for element
await waitForElement(() => screen.queryByText('Loading...'), 3000);

// Wait for API
await waitForApiCall(200);

// Wait for loading to finish
await waitForLoadingToFinish(() => screen.queryByText('Loading...'));

// Simulate typing
await typeWithDelay(inputElement, 'search query', 50);
```

## Overriding Handlers for Specific Tests

```typescript
import { server } from '../../test/mocks/server';
import { http, HttpResponse } from 'msw';

it('should handle server error', async () => {
  // Override handler for this test only
  server.use(
    http.get('http://localhost:8765/api/library/tracks', () => {
      return HttpResponse.json(
        { error: 'Server error' },
        { status: 500 }
      );
    })
  );

  // Test error handling
  const response = await fetch('http://localhost:8765/api/library/tracks');
  expect(response.status).toBe(500);
});
```

## Adding New Mock Handlers

Edit `src/test/mocks/handlers.ts`:

```typescript
// Add new handler
http.post(`${API_BASE}/my-new-endpoint`, async ({ request }) => {
  const body = await request.json();
  await delay(100);
  return HttpResponse.json({
    success: true,
    data: body
  });
}),
```

## Adding New Mock Data

Edit `src/test/mocks/mockData.ts`:

```typescript
export const mockMyData = Array.from({ length: 10 }, (_, i) => ({
  id: i + 1,
  name: `Item ${i + 1}`,
  // ... other fields
}));
```

## Common Test Patterns

### Testing Loading States
```typescript
it('should show loading state', async () => {
  render(<MyComponent />);

  // Loading should appear
  expect(screen.getByText('Loading...')).toBeInTheDocument();

  // Wait for data to load
  await waitFor(() => {
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
  });

  // Data should be visible
  expect(screen.getByText('Test Track 1')).toBeInTheDocument();
});
```

### Testing User Input
```typescript
it('should handle search input', async () => {
  const user = userEvent.setup();
  render(<SearchComponent />);

  const input = screen.getByRole('textbox');
  await user.type(input, 'test query');

  expect(input).toHaveValue('test query');
});
```

### Testing Error States
```typescript
it('should show error message', async () => {
  server.use(
    http.get('/api/tracks', () => {
      return HttpResponse.json({ error: 'Failed' }, { status: 500 });
    })
  );

  render(<MyComponent />);

  const error = await screen.findByText(/error/i);
  expect(error).toBeInTheDocument();
});
```

## Debugging Tests

### Enable Console Logs
```typescript
// In setup.ts, comment out console suppression:
// vi.spyOn(console, 'error').mockImplementation(() => {})
```

### Inspect MSW Requests
```typescript
// In your test
server.events.on('request:start', ({ request }) => {
  console.log('MSW intercepted:', request.method, request.url);
});
```

### Debug Test Output
```typescript
import { screen } from '@testing-library/react';

// Print current DOM
screen.debug();

// Print specific element
screen.debug(screen.getByRole('button'));
```

## Available Mock Endpoints

### Player
- `GET /api/player/state` - Get player state
- `POST /api/player/play` - Play track
- `POST /api/player/pause` - Pause
- `POST /api/player/seek` - Seek position
- `POST /api/player/volume` - Set volume
- `POST /api/player/next` - Next track
- `POST /api/player/previous` - Previous track

### Library
- `GET /api/library/tracks` - List tracks (pagination, search)
- `GET /api/library/tracks/:id` - Get single track
- `PUT /api/library/tracks/:id` - Update metadata
- `POST /api/library/tracks/:id/favorite` - Toggle favorite
- `GET /api/library/albums` - List albums
- `GET /api/library/artists` - List artists

### Enhancement
- `POST /api/player/enhancement/toggle` - Toggle enhancement
- `POST /api/player/enhancement/preset` - Set preset
- `POST /api/player/enhancement/intensity` - Set intensity

### Playlists
- `GET /api/playlists` - List playlists
- `POST /api/playlists` - Create playlist
- `GET /api/playlists/:id/tracks` - Get playlist tracks

See `src/test/mocks/handlers.ts` for complete list.

## Available Mock Data

- `mockTracks` - 100 tracks
- `mockAlbums` - 20 albums
- `mockArtists` - 10 artists
- `mockPlaylists` - 3 playlists
- `mockPlayerState` - Player state object
- `mockEnhancementPresets` - 5 presets

See `src/test/mocks/mockData.ts` for details.

## Troubleshooting

### Tests fail with "fetch is not defined"
MSW requires fetch API. Ensure setup.ts is imported in vitest.config.ts.

### Handlers not intercepting requests
- Check URL matches exactly (including protocol)
- Ensure server.listen() is called in beforeAll
- Verify handlers are exported from handlers.ts

### Timeout errors
Increase timeout in waitFor:
```typescript
await waitFor(() => { /* ... */ }, { timeout: 5000 });
```

## Best Practices

1. **One assertion per test** - Keep tests focused
2. **Use meaningful test names** - Describe what is tested
3. **Test user behavior** - Not implementation details
4. **Clean up after tests** - Automatic via afterEach
5. **Use realistic mock data** - Match production data structure
6. **Test error states** - Not just happy paths
7. **Avoid magic numbers** - Use named constants
8. **Group related tests** - Use describe blocks

## Resources

- [Testing Guidelines](/mnt/data/src/matchering/docs/development/TESTING_GUIDELINES.md)
- [Test Roadmap](/mnt/data/src/matchering/docs/development/TEST_IMPLEMENTATION_ROADMAP.md)
- [MSW Docs](https://mswjs.io/docs/)
- [React Testing Library](https://testing-library.com/react)
- [Vitest Docs](https://vitest.dev/)
