"""
Test and benchmark POMDP integration.

Verifies JAX/JIT compilation and validates POMDP behavior.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_scraping_pomdp():
    """Test Scraping POMDP initialization and inference."""
    print("\n" + "="*60)
    print("TESTING SCRAPING POMDP")
    print("="*60)

    from services.active_inference.scraping_pomdp import ScrapingPOMDP, JAX_AVAILABLE

    print(f"JAX Available: {JAX_AVAILABLE}")

    # Initialize POMDP
    start = time.time()
    pomdp = ScrapingPOMDP()
    init_time = time.time() - start
    print(f"Initialization time: {init_time*1000:.2f}ms")

    if pomdp.agent is None:
        print("WARNING: Agent not initialized (JAX unavailable)")
        return False

    # Test action selection (first call - JIT compilation)
    print("\nFirst action selection (includes JIT compilation):")
    start = time.time()
    result = pomdp.select_next_source()
    first_call = time.time() - start
    print(f"  Time: {first_call*1000:.2f}ms")
    print(f"  Source: {result['source']}")
    print(f"  Priority: {result.get('priority', 0):.3f}")
    print(f"  Method: {result.get('method', 'unknown')}")

    # Subsequent calls (cached JIT)
    print("\nSubsequent calls (JIT cached):")
    times = []
    for i in range(5):
        start = time.time()
        result = pomdp.select_next_source()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Call {i+1}: {elapsed*1000:.2f}ms - Source: {result['source']}")

    avg_time = np.mean(times)
    print(f"\nAverage inference time (cached): {avg_time*1000:.2f}ms")
    print(f"Speedup from JIT: {first_call/avg_time:.1f}x")

    # Test observation update
    print("\nTesting observation update:")
    update_result = pomdp.observe_scrape_result(
        source="skift",
        items_scraped=25,
        errors=0,
        novelty_ratio=0.8,
    )
    print(f"  Updated beliefs for 'skift': {update_result.get('beliefs', {})}")

    # Get status
    status = pomdp.get_status()
    print(f"\nPOMDP Status:")
    print(f"  Total observations: {status['total_observations']}")
    print(f"  Free energy: {status['free_energy']:.3f}")

    return True


def test_clustering_pomdp():
    """Test Clustering POMDP initialization and parameter selection."""
    print("\n" + "="*60)
    print("TESTING CLUSTERING POMDP")
    print("="*60)

    from services.active_inference.clustering_pomdp import ClusteringPOMDP, JAX_AVAILABLE

    print(f"JAX Available: {JAX_AVAILABLE}")

    # Initialize POMDP
    start = time.time()
    pomdp = ClusteringPOMDP()
    init_time = time.time() - start
    print(f"Initialization time: {init_time*1000:.2f}ms")

    if pomdp.agent is None:
        print("WARNING: Agent not initialized (JAX unavailable)")
        return False

    # Generate synthetic embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 384)  # Typical embedding dimension

    # Test parameter selection (first call - JIT compilation)
    print("\nFirst parameter selection (includes JIT compilation):")
    start = time.time()
    params = pomdp.select_parameters(embeddings)
    first_call = time.time() - start
    print(f"  Time: {first_call*1000:.2f}ms")
    print(f"  min_cluster_size: {params['min_cluster_size']}")
    print(f"  min_samples: {params['min_samples']}")
    print(f"  Confidence: {params.get('confidence', 0):.3f}")
    print(f"  Method: {params.get('method', 'unknown')}")

    # Subsequent calls
    print("\nSubsequent calls (JIT cached):")
    times = []
    for i in range(5):
        start = time.time()
        params = pomdp.select_parameters(embeddings)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Call {i+1}: {elapsed*1000:.2f}ms - Params: mcs={params['min_cluster_size']}, ms={params['min_samples']}")

    avg_time = np.mean(times)
    print(f"\nAverage inference time (cached): {avg_time*1000:.2f}ms")
    print(f"Speedup from JIT: {first_call/avg_time:.1f}x")

    # Test clustering observation
    print("\nTesting clustering result observation:")
    fake_labels = np.random.randint(-1, 5, size=100)
    result = pomdp.observe_clustering_result(
        params={"min_cluster_size": 3, "min_samples": 2},
        labels=fake_labels,
        embeddings=embeddings,
    )
    print(f"  Silhouette: {result['silhouette']:.3f}")
    print(f"  Num clusters: {result['num_clusters']}")
    print(f"  Noise ratio: {result['noise_ratio']:.3f}")

    return True


def test_extraction_pomdp():
    """Test Move Extraction POMDP initialization and method selection."""
    print("\n" + "="*60)
    print("TESTING MOVE EXTRACTION POMDP")
    print("="*60)

    from services.active_inference.move_extraction_pomdp import MoveExtractionPOMDP, JAX_AVAILABLE

    print(f"JAX Available: {JAX_AVAILABLE}")

    # Initialize POMDP
    start = time.time()
    pomdp = MoveExtractionPOMDP()
    init_time = time.time() - start
    print(f"Initialization time: {init_time*1000:.2f}ms")

    if pomdp.agent is None:
        print("WARNING: Agent not initialized (JAX unavailable)")
        return False

    # Test articles with different characteristics
    test_articles = [
        {
            "id": "1",
            "title": "Marriott Announces Major Expansion in Southeast Asia",
            "content": "Marriott International has announced a significant expansion..." * 50,
            "source": "skift",
        },
        {
            "id": "2",
            "title": "Hotel industry trends for 2024",
            "content": "The hotel industry continues to evolve..." * 20,
            "source": "reddit",
        },
        {
            "id": "3",
            "title": "random stuff",
            "content": "Some random content without much value.",
            "source": "unknown",
        },
        {
            "id": "4",
            "title": "Hilton Acquires New Property Portfolio for $500 Million",
            "content": "In a major deal, Hilton Hotels Corporation has acquired..." * 40,
            "source": "costar",
        },
    ]

    # Test method selection (first call - JIT compilation)
    print("\nFirst method selection (includes JIT compilation):")
    start = time.time()
    result = pomdp.select_extraction_method(test_articles[0])
    first_call = time.time() - start
    print(f"  Time: {first_call*1000:.2f}ms")
    print(f"  Method: {result['method']}")
    print(f"  Expected Quality: {result.get('expected_quality', 0):.3f}")
    print(f"  Expected Cost: {result.get('expected_cost', 0):.3f}")

    # Test all articles
    print("\nMethod selection for all test articles:")
    times = []
    for article in test_articles:
        start = time.time()
        result = pomdp.select_extraction_method(article)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  '{article['title'][:40]}...'")
        print(f"    Method: {result['method']}, Quality: {result.get('expected_quality', 0):.2f}, Time: {elapsed*1000:.2f}ms")

    avg_time = np.mean(times)
    print(f"\nAverage inference time (cached): {avg_time*1000:.2f}ms")
    print(f"Speedup from JIT: {first_call/avg_time:.1f}x")

    # Test extraction result observation
    print("\nTesting extraction result observation:")
    pomdp.observe_extraction_result(
        test_articles[0],
        "llm_single",
        {"confidence_score": 0.8, "company": "Marriott"},
    )
    pomdp.observe_extraction_result(
        test_articles[2],
        "skip",
        None,
    )

    status = pomdp.get_status()
    print(f"  Total extractions: {status['total_extractions']}")
    print(f"  LLM calls saved: {status['llm_calls_saved']}")
    print(f"  Savings rate: {status['savings_rate']:.1%}")

    return True


def benchmark_pomdp_inference():
    """Benchmark POMDP inference performance."""
    print("\n" + "="*60)
    print("BENCHMARK: POMDP INFERENCE PERFORMANCE")
    print("="*60)

    from services.active_inference.scraping_pomdp import ScrapingPOMDP, JAX_AVAILABLE

    if not JAX_AVAILABLE:
        print("JAX not available, skipping benchmark")
        return

    pomdp = ScrapingPOMDP()

    # Warm up JIT
    for _ in range(3):
        pomdp.select_next_source()

    # Benchmark
    n_iterations = 100
    times = []

    print(f"\nRunning {n_iterations} inference iterations...")
    for _ in range(n_iterations):
        start = time.time()
        pomdp.select_next_source()
        times.append(time.time() - start)

    times = np.array(times) * 1000  # Convert to ms

    print(f"\nResults ({n_iterations} iterations):")
    print(f"  Mean: {np.mean(times):.2f}ms")
    print(f"  Std: {np.std(times):.2f}ms")
    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Max: {np.max(times):.2f}ms")
    print(f"  P50: {np.percentile(times, 50):.2f}ms")
    print(f"  P95: {np.percentile(times, 95):.2f}ms")
    print(f"  P99: {np.percentile(times, 99):.2f}ms")

    # Check if JIT is working (subsequent calls should be much faster)
    if np.mean(times) < 200:  # Less than 200ms average
        print("\n[PASS] JIT compilation is working correctly!")
    else:
        print("\n[WARN] JIT may not be working optimally")


def test_integration():
    """Test integration with actual system components."""
    print("\n" + "="*60)
    print("TESTING SYSTEM INTEGRATION")
    print("="*60)

    # Test scheduler integration
    try:
        from scheduler.scheduler import ScraperScheduler
        scheduler = ScraperScheduler(use_pomdp=True)
        if scheduler.scraping_pomdp is not None:
            print("[OK] Scheduler POMDP integration: OK")
            result = scheduler.get_next_source_pomdp()
            print(f"  Next source recommendation: {result.get('source', 'N/A')}")
        else:
            print("[WARN] Scheduler POMDP: Not initialized")
    except Exception as e:
        print(f"[FAIL] Scheduler integration failed: {e}")

    # Test clustering integration
    try:
        from processing.clustering import ContentClusterer
        clusterer = ContentClusterer(use_adaptive=True)
        if clusterer.clustering_pomdp is not None:
            print("[OK] Clustering POMDP integration: OK")
        else:
            print("[WARN] Clustering POMDP: Not initialized")
    except Exception as e:
        print(f"[FAIL] Clustering integration failed: {e}")

    # Test hotelier bets integration
    try:
        from services.hotelier_bets import HotelierBetsService
        service = HotelierBetsService(use_adaptive=True)
        if service.extraction_pomdp is not None:
            print("[OK] Move Extraction POMDP integration: OK")
        else:
            print("[WARN] Move Extraction POMDP: Not initialized")
    except Exception as e:
        print(f"[FAIL] Hotelier Bets integration failed: {e}")


def main():
    """Run all tests."""
    print("="*60)
    print("POMDP INTEGRATION TESTS")
    print("="*60)

    results = {}

    # Test each POMDP
    try:
        results["scraping"] = test_scraping_pomdp()
    except Exception as e:
        print(f"Scraping POMDP test failed: {e}")
        results["scraping"] = False

    try:
        results["clustering"] = test_clustering_pomdp()
    except Exception as e:
        print(f"Clustering POMDP test failed: {e}")
        results["clustering"] = False

    try:
        results["extraction"] = test_extraction_pomdp()
    except Exception as e:
        print(f"Extraction POMDP test failed: {e}")
        results["extraction"] = False

    # Benchmark
    try:
        benchmark_pomdp_inference()
    except Exception as e:
        print(f"Benchmark failed: {e}")

    # Integration tests
    try:
        test_integration()
    except Exception as e:
        print(f"Integration test failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[PASS] All POMDP tests passed!")
    else:
        print("\n[WARN] Some tests failed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
