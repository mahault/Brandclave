#!/usr/bin/env python
"""Regenerate trends with quality filtering."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def main():
    from db.database import SessionLocal
    from db.models import TrendSignalModel
    from services.social_pulse import SocialPulseService

    print("Clearing old trends...")
    db = SessionLocal()
    count = db.query(TrendSignalModel).delete()
    db.commit()
    db.close()
    print(f"  Removed {count} old trends")

    print("Generating new trends with quality filtering...")
    service = SocialPulseService(days_back=30, use_llm=True, use_adaptive=True)
    trends = service.generate_trends(max_trends=20)

    print(f"  Generated {len(trends)} quality trends:")
    for i, t in enumerate(trends[:10], 1):
        name = t.get("name", "Unknown")[:50]
        score = t.get("strength_score", 0)
        print(f"    {i}. {name} ({score:.0%})")

    if len(trends) > 10:
        print(f"    ... and {len(trends) - 10} more")

    if trends:
        saved = service.save_trends(trends)
        print(f"  Saved {saved} trends to database")
    else:
        print("  No trends to save")

    return 0


if __name__ == "__main__":
    sys.exit(main())
