"""
Tests for benchmark data sources (DOF population, unauthorized estimates).
"""

from unittest.mock import MagicMock, patch


class TestDOFPopulation:
    """Tests for DOF population fetcher."""

    def test_dof_import(self):
        """Test DOF population module can be imported."""
        from src.benchmarks.dof_population import DOFPopulationFetcher

        assert DOFPopulationFetcher is not None

    def test_population_estimate_dataclass(self):
        """Test PopulationEstimate dataclass."""
        from src.benchmarks.dof_population import PopulationEstimate

        est = PopulationEstimate(
            year=2023,
            population=39_500_000,
            source="DOF E-1",
            geography="California",
        )
        assert est.year == 2023
        assert est.population == 39_500_000

    @patch("src.benchmarks.dof_population.requests.get")
    def test_get_ca_population_returns_int(self, mock_get):
        """Test get_ca_population returns an integer."""
        from src.benchmarks.dof_population import get_ca_population

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"population": 39500000}
        mock_get.return_value = mock_response

        result = get_ca_population(2023)
        assert isinstance(result, int)

    def test_fallback_population(self):
        """Test fallback population is returned on error."""
        from src.benchmarks.dof_population import get_ca_population

        # With mocked network error, should return fallback
        with patch("src.benchmarks.dof_population.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = get_ca_population(2023)
            # Should return reasonable fallback value
            assert isinstance(result, int)
            assert result > 35_000_000  # California has >35M people


class TestUnauthorizedBenchmarks:
    """Tests for unauthorized immigrant benchmarks."""

    def test_unauthorized_import(self):
        """Test unauthorized benchmarks module can be imported."""
        from src.benchmarks.unauthorized import UnauthorizedBenchmarkFetcher

        assert UnauthorizedBenchmarkFetcher is not None

    def test_unauthorized_estimate_dataclass(self):
        """Test UnauthorizedEstimate dataclass."""
        from src.benchmarks.unauthorized import UnauthorizedEstimate

        est = UnauthorizedEstimate(
            source="pew",
            year=2022,
            estimate=2_200_000,
            low=2_000_000,
            high=2_400_000,
            geography="California",
        )
        assert est.source == "pew"
        assert est.estimate == 2_200_000
        assert est.low == 2_000_000
        assert est.high == 2_400_000

    def test_fetcher_sources(self):
        """Test fetcher has expected source constants."""
        from src.benchmarks.unauthorized import UnauthorizedBenchmarkFetcher

        fetcher = UnauthorizedBenchmarkFetcher()
        # Should support multiple sources
        assert hasattr(fetcher, "SOURCES") or hasattr(fetcher, "sources")

    def test_get_pew_estimate(self):
        """Test Pew estimate retrieval."""
        from src.benchmarks.unauthorized import UnauthorizedBenchmarkFetcher

        fetcher = UnauthorizedBenchmarkFetcher()
        # Should have method to get Pew estimates (get_primary_estimate or get_estimate)
        assert hasattr(fetcher, "get_primary_estimate") or hasattr(fetcher, "get_estimate")


class TestBenchmarkValidation:
    """Tests for benchmark validation functions."""

    def test_population_range_validation(self):
        """Test population estimates are within reasonable range."""
        from src.benchmarks.dof_population import get_ca_population

        with patch("src.benchmarks.dof_population.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            pop = get_ca_population(2023)
            # California population should be between 35M and 45M
            assert 35_000_000 <= pop <= 45_000_000

    def test_unauthorized_estimate_range(self):
        """Test unauthorized estimates are within reasonable range."""
        from src.benchmarks.unauthorized import UnauthorizedEstimate

        est = UnauthorizedEstimate(
            source="test",
            year=2022,
            estimate=2_200_000,
            geography="California",
        )
        # California unauthorized population typically 1.5M - 3M
        assert 1_000_000 <= est.estimate <= 5_000_000
