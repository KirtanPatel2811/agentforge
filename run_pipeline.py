"""
run_pipeline.py — CLI runner for AgentForge
Run: python run_pipeline.py
Or:  python run_pipeline.py "Your custom goal here"
"""

import sys
import time

# Allow passing goal as command-line argument
if len(sys.argv) > 1:
    GOAL = " ".join(sys.argv[1:])
else:
    GOAL = (
        "Find the top 5 countries by solar energy installed capacity "
        "and write a brief summary with key statistics and sources"
    )

print("\n" + "=" * 65)
print("  AgentForge — Multi-Agent Collaborative Task System")
print("=" * 65)
print(f"\nGoal: {GOAL}\n")
print("Starting pipeline...\n")

start = time.time()

from src.agents.planner import PlannerAgent

planner = PlannerAgent()
result = planner.run_goal(GOAL)

elapsed = time.time() - start

print("\n" + "=" * 65)
print("  FINAL REPORT")
print("=" * 65)
print(result["report"])

print("\n" + "=" * 65)
print("  CRITIC ASSESSMENT")
print("=" * 65)
print(f"  Score    : {result['critic_score']:.2f} / 1.00")
print(f"  Approved : {result['approved']}")
print(f"  Task ID  : {result['task_id']}")
print(f"  Runtime  : {elapsed:.1f}s")

if result["critic_feedback"]:
    print(f"  Feedback : {result['critic_feedback'][:200]}")

print("\n" + "=" * 65)
print(f"  Done! Report saved to data/outputs/ if charts were generated.")
print("=" * 65 + "\n")

# Usage examples printed if no custom goal provided
if len(sys.argv) == 1:
    print("Tip: Pass a custom goal as an argument:")
    print('  python run_pipeline.py "Compare AI chip manufacturers: Nvidia vs AMD vs Intel"')
    print('  python run_pipeline.py "Research LSTM vs Transformer for time series forecasting"')
    print('  python run_pipeline.py "Find top 5 AI papers on ArXiv this month and summarise them"')
    print()
"""
