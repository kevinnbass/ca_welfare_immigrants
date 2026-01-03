"""
Integration tests for SNAP estimand weight verification.

CRITICAL: These tests verify that SNAP rates use the correct weights:
- Household rates: WGTP (household weights)
- Person rates: PWGTP (person weights)
"""


class TestSNAPEstimandWeights:
    """Tests verifying SNAP uses correct weights based on unit of analysis."""

    def test_snap_household_householder_uses_wgtp(self):
        """CRITICAL: Verify SNAP household (householder) rate uses WGTP weights."""
        from src.utils.analysis_units import get_estimand_spec

        spec = get_estimand_spec("snap_household_householder")

        assert spec.weight_col == "WGTP", (
            f"SNAP household (householder) should use WGTP, got {spec.weight_col}"
        )
        assert spec.weight_prefix == "WGTP", (
            f"SNAP household (householder) weight prefix should be WGTP, got {spec.weight_prefix}"
        )
        assert spec.unit == "household", (
            f"SNAP household estimand unit should be 'household', got {spec.unit}"
        )

    def test_snap_household_highest_risk_uses_wgtp(self):
        """CRITICAL: Verify SNAP household (highest-risk) rate uses WGTP weights."""
        from src.utils.analysis_units import get_estimand_spec

        spec = get_estimand_spec("snap_household_highest_risk")

        assert spec.weight_col == "WGTP", (
            f"SNAP household (highest-risk) should use WGTP, got {spec.weight_col}"
        )
        assert spec.weight_prefix == "WGTP", (
            f"SNAP household (highest-risk) weight prefix should be WGTP, got {spec.weight_prefix}"
        )
        assert spec.unit == "household", (
            f"SNAP household estimand unit should be 'household', got {spec.unit}"
        )

    def test_snap_person_uses_pwgtp(self):
        """CRITICAL: Verify SNAP person rate uses PWGTP weights."""
        from src.utils.analysis_units import get_estimand_spec

        spec = get_estimand_spec("snap_person")

        assert spec.weight_col == "PWGTP", f"SNAP person should use PWGTP, got {spec.weight_col}"
        assert spec.weight_prefix == "PWGTP", (
            f"SNAP person weight prefix should be PWGTP, got {spec.weight_prefix}"
        )
        assert spec.unit == "person", (
            f"SNAP person estimand unit should be 'person', got {spec.unit}"
        )


class TestMedicaidEstimandWeights:
    """Tests verifying Medicaid uses correct weights."""

    def test_medicaid_uses_pwgtp(self):
        """Verify Medicaid rate uses person weights."""
        from src.utils.analysis_units import get_estimand_spec

        spec = get_estimand_spec("medicaid")

        assert spec.weight_col == "PWGTP", f"Medicaid should use PWGTP, got {spec.weight_col}"
        assert spec.unit == "person", f"Medicaid estimand unit should be 'person', got {spec.unit}"


class TestSSIEstimandWeights:
    """Tests verifying SSI uses correct weights."""

    def test_ssi_uses_pwgtp(self):
        """Verify SSI rate uses person weights."""
        from src.utils.analysis_units import get_estimand_spec

        spec = get_estimand_spec("ssi")

        assert spec.weight_col == "PWGTP", f"SSI should use PWGTP, got {spec.weight_col}"
        assert spec.unit == "person", f"SSI estimand unit should be 'person', got {spec.unit}"


class TestEstimandSpecConsistency:
    """Tests for estimand specification consistency."""

    def test_household_estimands_have_hh_status_rule(self):
        """Verify household estimands have status assignment rule."""
        from src.utils.analysis_units import get_estimand_spec

        for estimand_id in ["snap_household_householder", "snap_household_highest_risk"]:
            spec = get_estimand_spec(estimand_id)
            assert spec.hh_status_rule is not None, (
                f"Household estimand {estimand_id} should have hh_status_rule"
            )

    def test_person_estimands_have_no_hh_status_rule(self):
        """Verify person estimands don't have status assignment rule."""
        from src.utils.analysis_units import get_estimand_spec

        for estimand_id in ["snap_person", "medicaid", "ssi"]:
            spec = get_estimand_spec(estimand_id)
            assert spec.hh_status_rule is None, (
                f"Person estimand {estimand_id} should not have hh_status_rule"
            )

    def test_replicate_weight_prefix_consistency(self):
        """Verify replicate weight prefix matches main weight column."""
        from src.utils.analysis_units import get_estimand_spec

        estimands = [
            "medicaid",
            "ssi",
            "snap_person",
            "snap_household_householder",
            "snap_household_highest_risk",
        ]

        for estimand_id in estimands:
            spec = get_estimand_spec(estimand_id)
            # Weight prefix should be base of weight column
            assert spec.weight_prefix in spec.weight_col, (
                f"{estimand_id}: weight_prefix '{spec.weight_prefix}' not in weight_col '{spec.weight_col}'"
            )


class TestWelfarePrograms:
    """Tests for WELFARE_PROGRAMS configuration."""

    def test_welfare_programs_have_weight_specs(self):
        """Verify all welfare programs have weight specifications."""
        from src import config

        required_fields = ["variable", "weight_col", "weight_prefix", "unit"]

        for program, spec in config.WELFARE_PROGRAMS.items():
            if program in ["snap"]:  # Legacy key may not have all fields
                continue

            for field in required_fields:
                assert field in spec, f"Program '{program}' missing required field '{field}'"

    def test_snap_variants_exist(self):
        """Verify SNAP has household and person variants."""
        from src import config

        expected_variants = [
            "snap_household_householder",
            "snap_household_highest_risk",
            "snap_person",
        ]

        for variant in expected_variants:
            assert variant in config.WELFARE_PROGRAMS, f"Missing SNAP variant: {variant}"
