import numpy as np
import cv2
import mnist

# ① 훈련 데이터와 테스트 데이터 로드 ---
train, train_labels = mnist.getTrain()
test, test_labels = mnist.getTest()

print(f"훈련 데이터: {train.shape[0]}개 샘플")
print(f"테스트 데이터: {test.shape[0]}개 샘플")

# ② k-NN 모델 생성 및 훈련 ---
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

print("✅ k-NN 모델 훈련 완료")

# ③ k값을 1~10까지 변경하며 정확도 측정 ---
accuracies = {}

for k in range(1, 21):
    ret, result, neighbors, distance = knn.findNearest(test, k=k)
    correct = np.sum(result == test_labels)
    accuracy = correct / result.size * 100.0
    accuracies[k] = accuracy

    print(f"k={k:2d}: {accuracy:.2f}%")

# 최적 k값 찾기
best_k = max(accuracies, key=accuracies.get)
best_acc = accuracies[best_k]
print(f"\n✅ 최적 k값: {best_k} (정확도: {best_acc:.2f}%)")

