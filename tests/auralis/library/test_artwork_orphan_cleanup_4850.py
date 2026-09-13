# -*- coding: utf-8 -*-
"""
Re-extracting/re-downloading artwork unlinks the superseded file (#4850)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`delete_artwork` correctly removes the on-disk file before clearing the DB
column. The other two write paths -- `extract_and_save_artwork` (backing
`POST /artwork/extract`) and `update_artwork_path` (backing
`POST /artwork/download`) -- only ever overwrote `album.artwork_path`,
leaving the old file on disk. Since every write generates a unique
`album_{id}_{content_hash}.ext` filename, every re-extract/re-download that
produced different bytes leaked one more permanent orphan under
`~/.auralis/artwork`.

These tests swap `AlbumRepository.artwork_extractor` for one rooted at
`tmp_path` (its real constructor always points at `~/.auralis/artwork`) so
nothing here touches the real user artwork cache, but keep the *real*
`ArtworkExtractor.delete_artwork` so the actual unlink is exercised, not a
mock standing in for it.
"""

from unittest.mock import MagicMock

from auralis.library.artwork import ArtworkExtractor
from auralis.library.models import Album


def _seed_album_with_artwork(album_repository, title: str, artwork_path: str | None) -> int:
    """Create an Album row with a pre-existing artwork_path, return its id."""
    with album_repository._session_scope() as session:
        album = Album(title=title, artwork_path=artwork_path)
        session.add(album)
        session.commit()
        return album.id


class TestUpdateArtworkPathUnlinksThePrevious:
    """The download flow's write path (`update_artwork_path`)."""

    def test_unlinks_the_superseded_file_after_a_successful_commit(self, album_repository, tmp_path):
        album_repository.artwork_extractor = ArtworkExtractor(str(tmp_path))

        old_file = tmp_path / "album_1_old.jpg"
        old_file.write_bytes(b"old-cover")
        new_file = tmp_path / "album_1_new.jpg"
        new_file.write_bytes(b"new-cover")

        album_id = _seed_album_with_artwork(album_repository, "Download Test", str(old_file))

        updated = album_repository.update_artwork_path(album_id, str(new_file))

        assert updated is not None
        assert not old_file.exists(), "the superseded artwork file must be unlinked"
        assert new_file.exists()

    def test_does_not_delete_when_the_path_is_unchanged(self, album_repository, tmp_path):
        """A re-save that hashes to the same filename must not delete it."""
        album_repository.artwork_extractor = ArtworkExtractor(str(tmp_path))

        same_file = tmp_path / "album_2_same.jpg"
        same_file.write_bytes(b"cover")

        album_id = _seed_album_with_artwork(album_repository, "Same Path Test", str(same_file))

        updated = album_repository.update_artwork_path(album_id, str(same_file))

        assert updated is not None
        assert same_file.exists()

    def test_no_previous_artwork_is_a_no_op_not_an_error(self, album_repository, tmp_path):
        album_repository.artwork_extractor = ArtworkExtractor(str(tmp_path))
        new_file = tmp_path / "album_3_new.jpg"
        new_file.write_bytes(b"cover")

        album_id = _seed_album_with_artwork(album_repository, "No Previous Test", None)

        updated = album_repository.update_artwork_path(album_id, str(new_file))

        assert updated is not None
        assert new_file.exists()


class TestExtractAndSaveArtworkUnlinksThePrevious:
    """The extract flow's write path (`extract_and_save_artwork`)."""

    def _extractor_returning(self, tmp_path, new_path: str) -> ArtworkExtractor:
        extractor = ArtworkExtractor(str(tmp_path))
        extractor.extract_artwork = MagicMock(return_value=new_path)
        return extractor

    def _seed_album_with_track(self, album_repository, track_repository, tmp_path, title: str):
        track_file = tmp_path / "track.flac"
        track_file.write_bytes(b"\x00")
        track = track_repository.add({
            'title': 'T',
            'filepath': str(track_file),
            'duration': 1.0,
            'sample_rate': 44100,
            'channels': 2,
            'format': 'FLAC',
            # _get_or_create_album only creates the album when an artist_id
            # resolves (see track_repository_lifecycle.py), so an artist is
            # required here even though this test doesn't otherwise care
            # about it.
            'artists': ['Orphan Test Artist'],
            'album': title,
        })
        assert track is not None
        album = album_repository.get_by_title(title)
        assert album is not None
        return album.id

    def test_unlinks_the_superseded_file_after_a_successful_commit(
        self, album_repository, track_repository, tmp_path
    ):
        old_file = tmp_path / "album_x_old.jpg"
        old_file.write_bytes(b"old-cover")
        new_file = tmp_path / "album_x_new.jpg"
        new_file.write_bytes(b"new-cover")

        album_repository.artwork_extractor = self._extractor_returning(tmp_path, str(new_file))
        album_id = self._seed_album_with_track(
            album_repository, track_repository, tmp_path, "Extract Test"
        )
        with album_repository._session_scope() as session:
            session.get(Album, album_id).artwork_path = str(old_file)
            session.commit()

        result = album_repository.extract_and_save_artwork(album_id)

        assert result == str(new_file)
        assert not old_file.exists(), "the superseded artwork file must be unlinked"
        assert new_file.exists()

    def test_does_not_delete_when_the_path_is_unchanged(
        self, album_repository, track_repository, tmp_path
    ):
        same_file = tmp_path / "album_y_same.jpg"
        same_file.write_bytes(b"cover")

        album_repository.artwork_extractor = self._extractor_returning(tmp_path, str(same_file))
        album_id = self._seed_album_with_track(
            album_repository, track_repository, tmp_path, "Extract Same Path Test"
        )
        with album_repository._session_scope() as session:
            session.get(Album, album_id).artwork_path = str(same_file)
            session.commit()

        result = album_repository.extract_and_save_artwork(album_id)

        assert result == str(same_file)
        assert same_file.exists()

    def test_no_previous_artwork_is_a_no_op_not_an_error(
        self, album_repository, track_repository, tmp_path
    ):
        new_file = tmp_path / "album_z_new.jpg"
        new_file.write_bytes(b"cover")

        album_repository.artwork_extractor = self._extractor_returning(tmp_path, str(new_file))
        album_id = self._seed_album_with_track(
            album_repository, track_repository, tmp_path, "Extract No Previous Test"
        )

        result = album_repository.extract_and_save_artwork(album_id)

        assert result == str(new_file)
        assert new_file.exists()
