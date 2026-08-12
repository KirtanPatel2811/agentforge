# run_pipeline.py
from src.agents.planner import PlannerAgent

planner = PlannerAgent()

result = planner.run_goal(
    "Find the top 5 countries by solar energy capacity "
    "and write a brief summary with key statistics"
)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(result["report"][:2000])

print("\n" + "=" * 60)
print("CRITIC ASSESSMENT")
print("=" * 60)
print("Score   :", round(result["critic_score"], 2))
print("Approved:", result["approved"])

if result["critic_feedback"]:
    print("Feedback:", result["critic_feedback"][:300])
