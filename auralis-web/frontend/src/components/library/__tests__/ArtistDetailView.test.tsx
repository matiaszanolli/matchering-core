/**
 * ArtistDetailView Component Tests
 *
 * Behaviour-driven tests covering data display, loading/error states,
 * navigation callbacks, and playback interactions.
 */

import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { render, screen } from '@/test/test-utils';
import ArtistDetailView from '../Details/ArtistDetailView';

// Mock the data hook — this is the sole data source for the component
const mockUseArtistDetailsData = vi.fn();
vi.mock('../Details/useArtistDetailsData', () => ({
  useArtistDetailsData: (...args: unknown[]) => mockUseArtistDetailsData(...args),
}));

// #3940: playback now goes through the leaf-level usePlayTrack hook instead of a
// drilled onTrackPlay prop. Mock it so we can assert what the view plays.
const mockPlayTrack = vi.fn();
vi.mock('@/hooks/player', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/player')>()),
  usePlayTrack: () => ({ playTrack: mockPlayTrack }),
}));

// Mock sub-components to isolate ArtistDetailView logic
vi.mock('../Details/DetailLoading', () => ({
  default: () => <div data-testid="detail-loading">Loading...</div>,
}));

vi.mock('../Details/ArtistDetailHeader', () => ({
  ArtistDetailHeaderSection: ({ artist, onBack, onPlayAll, onShuffle }: any) => (
    <div data-testid="artist-header">
      <span data-testid="artist-name">{artist.name}</span>
      {onBack && <button data-testid="back-btn" onClick={onBack}>Back</button>}
      <button data-testid="play-all-btn" onClick={onPlayAll}>Play All</button>
      <button data-testid="shuffle-btn" onClick={onShuffle}>Shuffle</button>
    </div>
  ),
}));

vi.mock('../Details/ArtistDetailTabs', () => ({
  ArtistDetailTabsSection: ({ artist, onTrackClick, onAlbumClick }: any) => (
    <div data-testid="artist-tabs">
      {artist.albums?.map((a: any) => (
        <button key={a.id} data-testid={`album-${a.id}`} onClick={() => onAlbumClick(a.id)}>
          {a.title}
        </button>
      ))}
      {artist.tracks?.map((t: any) => (
        <button key={t.id} data-testid={`track-${t.id}`} onClick={() => onTrackClick(t)}>
          {t.title}
        </button>
      ))}
    </div>
  ),
}));

const mockArtist = {
  id: 1,
  name: 'Queen',
  albums: [
    { id: 10, title: 'A Night at the Opera' },
    { id: 11, title: 'News of the World' },
  ],
  tracks: [
    { id: 100, title: 'Bohemian Rhapsody', duration: 354 },
    { id: 101, title: 'We Will Rock You', duration: 122 },
    { id: 102, title: "Don't Stop Me Now", duration: 210 },
  ],
};

describe('ArtistDetailView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Loading state', () => {
    it('renders loading indicator while data is being fetched', () => {
      mockUseArtistDetailsData.mockReturnValue({ artist: null, loading: true, error: null });
      render(<ArtistDetailView artistId={1} />);
      expect(screen.getByTestId('detail-loading')).toBeInTheDocument();
    });
  });

  describe('Error state (#4643)', () => {
    it('renders error message when fetch fails', () => {
      mockUseArtistDetailsData.mockReturnValue({
        artist: null, loading: false, error: { status: 0, message: 'Network error' },
      });
      render(<ArtistDetailView artistId={1} />);
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('renders "Artist not found" when artist is null without error', () => {
      mockUseArtistDetailsData.mockReturnValue({ artist: null, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('renders "Artist not found" (not a generic error) for a 404', () => {
      mockUseArtistDetailsData.mockReturnValue({
        artist: null, loading: false, error: { status: 404, message: 'Artist 999 not found' },
      });
      render(<ArtistDetailView artistId={999} />);
      expect(screen.getByText('Artist not found')).toBeInTheDocument();
      expect(screen.queryByText('Error Loading Artist')).not.toBeInTheDocument();
    });

    it('renders "Error Loading Artist" with the message for a 500 — distinguishable from a 404', () => {
      mockUseArtistDetailsData.mockReturnValue({
        artist: null, loading: false, error: { status: 500, message: 'Internal server error' },
      });
      render(<ArtistDetailView artistId={1} />);
      expect(screen.getByText('Error Loading Artist')).toBeInTheDocument();
      expect(screen.getByText('Internal server error')).toBeInTheDocument();
      expect(screen.queryByText('Artist not found')).not.toBeInTheDocument();
    });
  });

  describe('Data display', () => {
    it('renders artist name in header', () => {
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      expect(screen.getByTestId('artist-name')).toHaveTextContent('Queen');
    });

    it('renders albums and tracks in tabs section', () => {
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      expect(screen.getByTestId('album-10')).toHaveTextContent('A Night at the Opera');
      expect(screen.getByTestId('track-100')).toHaveTextContent('Bohemian Rhapsody');
    });

    it('passes artistId to the data hook', () => {
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={42} />);
      expect(mockUseArtistDetailsData).toHaveBeenCalledWith(42);
    });
  });

  describe('Callbacks', () => {
    it('calls onBack when back button clicked', async () => {
      const user = userEvent.setup();
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      const onBack = vi.fn();
      render(<ArtistDetailView artistId={1} onBack={onBack} />);
      await user.click(screen.getByTestId('back-btn'));
      expect(onBack).toHaveBeenCalledOnce();
    });

    it('plays the first track on Play All', async () => {
      const user = userEvent.setup();
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      await user.click(screen.getByTestId('play-all-btn'));
      expect(mockPlayTrack).toHaveBeenCalledWith(mockArtist.tracks[0]);
    });

    it('plays a track on Shuffle', async () => {
      const user = userEvent.setup();
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      await user.click(screen.getByTestId('shuffle-btn'));
      expect(mockPlayTrack).toHaveBeenCalledOnce();
      // The argument should be one of the tracks
      const calledWith = mockPlayTrack.mock.calls[0][0];
      expect(mockArtist.tracks).toContainEqual(calledWith);
    });

    it('calls onAlbumClick when album is clicked', async () => {
      const user = userEvent.setup();
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      const onAlbumClick = vi.fn();
      render(<ArtistDetailView artistId={1} onAlbumClick={onAlbumClick} />);
      await user.click(screen.getByTestId('album-10'));
      expect(onAlbumClick).toHaveBeenCalledWith(10);
    });

    it('plays the track when a track is clicked', async () => {
      const user = userEvent.setup();
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      await user.click(screen.getByTestId('track-101'));
      expect(mockPlayTrack).toHaveBeenCalledWith(mockArtist.tracks[1]);
    });

    it('does not crash when optional callbacks are omitted', async () => {
      const user = userEvent.setup();
      mockUseArtistDetailsData.mockReturnValue({ artist: mockArtist, loading: false, error: null });
      render(<ArtistDetailView artistId={1} />);
      // Play All / Shuffle should not throw
      await user.click(screen.getByTestId('play-all-btn'));
      await user.click(screen.getByTestId('shuffle-btn'));
    });
  });
});
