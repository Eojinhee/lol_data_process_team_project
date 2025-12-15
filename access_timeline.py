import json

with open("match_0.json", "r", encoding="utf-8") as f:
    matches = json.load(f)

with open("timeline_0.json", "r", encoding="utf-8") as f:
    timelines = json.load(f)


match = matches[0]['info']
timeline = timelines[0]['info']

first_p = match['participants'][0]
first_pid = first_p['participantId']
print(f"챔피언 : {first_p['championName']}, pid : {first_pid}")

# 14분 시점의 누적 골드량을 알고 싶다
minutes = 14
to_extract = "totalGold"
single_frame = timeline["frames"][minutes]

target_time = single_frame["timestamp"] / 60000 # 분
score = single_frame["participantFrames"][str(first_pid)][to_extract]

print(f"{target_time:.2f} 분에서 {to_extract} 값: {score}")