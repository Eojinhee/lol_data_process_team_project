import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_failed_matches():
  """
  '테스트 데이터' 중 예측에 실패한 경기들의 상세 지표를 시각화하고 분석합니다.
  """
  try:
    # [수정] 읽어오는 파일명을 테스트 데이터용 파일로 변경
    predictions_df = pd.read_csv('test_prediction_results.csv')
    features_df = pd.read_csv('test_match_features.csv')
  except FileNotFoundError:
    print("오류: 'test_prediction_results.csv' 또는 'test_match_features.csv' 파일을 찾을 수 없습니다.")
    print("먼저 메인 분석 스크립트를 실행하여 두 파일을 생성해주세요.")
    return

  # 두 데이터프레임을 matchId 기준으로 병합
  merged_df = pd.merge(predictions_df, features_df, on='matchId')

  # 예측에 실패한 경기만 필터링
  failed_matches = merged_df[merged_df['actual_win'] != merged_df['predicted_win']].copy()

  if failed_matches.empty:
    print("\n🎉 테스트 데이터 내 예측 실패 경기가 없습니다! 모델이 완벽합니다.")
    return

  print(f"\n총 {len(failed_matches)}개의 테스트 데이터 예측 실패 경기에 대한 상세 분석을 시작합니다.")

  # 피처 이름 정의
  feature_names = {
    'TOP': ['골드 가속도', '안정적 성장', '타워 대미지', '1v1 교전', '시야 점수'],
    'JUNGLE': ['오브젝트 컨트롤', '갱킹 성공률', '경험치 격차', '정글링 효율', '시야 점수'],
    'MIDDLE': ['초반 자원 우위', '맵 장악 주도권', '1v1 교전', '퍼블 영향력', '생존/압박'],
    'DUO_APT': ['듀오 킬 시너지', 'ADC 보호 효율', '시야-오브젝트 전환', '성장-압박 전환', '교전 집중도']
  }

  # Mac 환경에 맞는 한글 폰트 설정
  plt.rcParams['font.family'] = 'AppleGothic'
  plt.rcParams['axes.unicode_minus'] = False

  # 실패한 경기들을 하나씩 순회하며 시각화
  for index, row in failed_matches.iterrows():
    match_id = row['matchId']
    actual_winner = "블루팀" if row['actual_win'] == 1 else "레드팀"
    predicted_winner = "블루팀" if row['predicted_win'] == 1 else "레드팀"

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"실패 분석 (Test Set) | 경기 ID: {match_id}\n(실제 승리: {actual_winner}, 모델 예측: {predicted_winner})",
                 fontsize=22, y=0.98)

    print("\n" + "=" * 50)
    print(f"분석 경기: {match_id} (실제: {actual_winner} 승 / 예측: {predicted_winner} 승)")

    analysis_text = f"모델은 블루팀의 평균 승리 확률({row['blue_win_probability']:.1%})을 기반으로 {predicted_winner}의 승리를 예측했습니다.\n"

    for i, lane in enumerate(feature_names.keys()):
      ax = axes[i // 2, i % 2]
      lane_feature_keys = [f"{lane}_{j}" for j in range(5)]
      values = row[lane_feature_keys].fillna(0).values
      names = feature_names[lane]

      colors = ['#5A9CFF' if v >= 0 else '#FF5A5A' for v in values]
      sns.barplot(x=values, y=names, ax=ax, palette=colors, orient='h')

      ax.set_title(f'{lane} 라인 지표 (블루팀 - 레드팀)', fontsize=15)
      ax.set_xlabel('지표 값 (양수: 블루 우세, 음수: 레드 우세)', fontsize=12)
      ax.set_ylabel('')
      ax.axvline(0, color='grey', linestyle='--')

      if actual_winner == "레드팀" and np.min(values) < 0:
        worst_feature_idx = np.argmin(values)
        analysis_text += f"실제로는 [{lane}] 라인의 '{names[worst_feature_idx]}' 지표({values[worst_feature_idx]:.2f})에서 크게 밀린 것이 역전의 빌미가 되었을 수 있습니다.\n"
      elif actual_winner == "블루팀" and np.max(values) > 0:
        best_feature_idx = np.argmax(values)
        analysis_text += f"실제로는 [{lane}] 라인의 '{names[best_feature_idx]}' 지표({values[best_feature_idx]:.2f})에서 압도적인 우위를 점하며 승리했을 가능성이 있습니다.\n"

    print(analysis_text)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
  analyze_failed_matches()
