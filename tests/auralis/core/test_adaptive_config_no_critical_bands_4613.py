"""AdaptiveConfig no longer advertises a critical-band count it ignores (#4613).

`AdaptiveConfig.critical_bands` was validated to 8-64 (default 26) but never
read: `PsychoacousticEQ` always builds the fixed 25-band Bark table via
`create_critical_bands()`, which takes no arguments. The range assert made the
knob look functional.
"""

import dataclasses

from auralis.core.config.settings import AdaptiveConfig
from auralis.core.config.unified_config import UnifiedConfig
from auralis.dsp.eq.critical_bands import create_critical_bands


def test_the_field_is_gone():
    assert "critical_bands" not in {f.name for f in dataclasses.fields(AdaptiveConfig)}


def test_the_band_layout_is_the_fixed_25_band_table():
    assert len(create_critical_bands()) == 25


def test_unified_config_still_round_trips():
    """to_dict/from_dict never carried the field; removing it must not break
    a config built from its own serialization."""
    original = UnifiedConfig()
    rebuilt = UnifiedConfig.from_dict(original.to_dict())
    assert rebuilt.adaptive.mode == original.adaptive.mode
    assert rebuilt.adaptive.adaptation_strength == original.adaptive.adaptation_strength
