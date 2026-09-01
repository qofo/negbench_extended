수정본은 이전 버전보다 분명히 좋아졌습니다. 특히 제가 이전 리뷰에서 문제 삼았던 “수학적 분해 자체를 novelty처럼 보이게 하는 문제”와 “alignment intervention의 원리적 한계”를 상당 부분 뒤로 물렸습니다. 다만 새 버전에는 새로운 통계적·논리적 문제가 몇 가지 생겼고, 아직도 핵심 주장 중 일부는 증거보다 강합니다.

현재 수정본을 기준으로 다시 리뷰하면, 저는 여전히 **Borderline / Weak Reject와 Weak Accept 사이**에 두겠습니다. 이전 버전보다는 확실히 올라갔지만, 아직 “이 논문을 받아야 하는 결정적 이유”가 완전히 만들어진 것은 아닙니다.

## 1. 가장 크게 개선된 점

### 1) novelty의 중심을 제대로 옮겼습니다

가장 좋은 수정입니다.

이전에는 2×2 decomposition 자체가 상당히 전면에 있었는데, 이제는 명시적으로

> “분해 자체는 2×2 요인 설계의 표준 도구이며 기여가 아니다. 기여는 그 도구로 얻은 실측이다”

라고 했습니다. 

이것은 반드시 필요한 수정이었습니다.

이제 논문의 novelty는

$$
\text{representation에서 정보가 검출됨}
$$

과

$$
\text{그 정보가 similarity ranking을 지배함}
$$

을 구분하고, 실제로

$$
\gamma \ll |\alpha|,|\beta|
$$

임을 측정했다는 쪽으로 이동했습니다.

이 framing은 훨씬 방어하기 쉽습니다.

---

### 2) “정보가 없다”는 기존 연구와 정면충돌하지 않도록 바뀌었습니다

서론에서

> “이 결과들은 부정 관련 정보가 CLIP 표현에서 회복되거나 조작될 수 있음을 보인다. 그러나 그것이 최종 크로스모달 인터페이스에서 순위를 결정할 만큼 강하다는 것까지 확립하지는 않는다.”

라고 한 부분은 상당히 좋습니다. 

이렇게 하면 기존 연구의 결론을 부정하는 대신, 그 연구가 측정하지 않은 것을 측정한다고 주장할 수 있습니다.

특히 최근 연구들이 negation direction, subspace, intermediate feature 등에 대해 정보를 찾고 있다는 점을 고려하면 이 포지셔닝이 적절합니다. 예를 들어 Quantmeyer et al.은 CLIP 내부에서 negation processing을 localization했고, 후속 연구들은 representation-level intervention을 사용합니다.

---

### 3) AB-swap과 positional shortcut을 추가한 것은 좋은 수정입니다

이 부분도 이전보다 상당히 좋아졌습니다.

단순히

> “text probe가 73.8%니까 negation 정보가 있다”

라고 하면 너무 쉽게 공격받습니다.

그런데 현재는 어순 shortcut을 검사해서

$$
73.80\%\rightarrow64.69\%
$$

로 떨어지는 것을 보고하고, 그래도 25% pairwise chance보다 높으며 template transfer가 95% 유지된다고 했습니다. 

즉 “프로브가 단순히 no가 어디 있는지를 읽는 것”이라는 반론을 어느 정도 차단했습니다.

이건 이전 버전보다 명확한 개선입니다.

---

### 4) 16.7% chance를 그대로 사용하지 않고 cross-concept control을 추가한 것은 방향이 좋습니다

이전 버전의 가장 약한 부분 중 하나가 “4개 score의 random ordering이므로 16.7%”를 실제 baseline처럼 사용하는 것이었습니다.

이번에는

> “기준선을 가정하지 않고 실측한다.”

고 바꾸고, 서로 다른 개념의 이미지와 텍스트를 결합해서 0.67%라는 empirical control을 추가했습니다. 

이 아이디어 자체는 좋습니다.

다만 아래에서 설명하겠지만 **현재 그 결과의 해석은 아직 잘못되어 있습니다.**

---

### 5) 기존 intervention에 대한 과격한 upper-bound 주장을 상당 부분 제거했습니다

이건 매우 중요합니다.

이전 버전의

> “alignment를 아무리 개선해도 최대 4.6배”

같은 주장은 일반적인 \(W\)에 대해 성립하지 않았습니다.

이번 버전에서는 그 위험한 논리를 제거하고

$$
S_W=v^\top Wt
$$

를 직접 학습하여 rank-32 transformation이 29.19%에 도달한다고 실험적으로 보여줍니다. 

이쪽이 훨씬 낫습니다.

“기존 방법은 원리적으로 불가능하다”가 아니라

> “동일한 embedding에서 similarity rule만 바꿔도 상당한 개선이 가능하다”

라는 empirical claim이 되었기 때문입니다.

---

# 2. 아직도 잘못되었거나 위험한 부분

여기부터가 중요합니다. 현재 논문에서 제가 리뷰어라면 실제로 수정 요구를 할 부분입니다.

## 2-1. 가장 큰 문제: “교차항이 객체 특이적 의미 상호작용”이라는 결론은 아직 과합니다

논문은

> γ가 42개 개념 전부에서 0보다 유의하게 크고
> cross-concept control에서도 실제 γ가 더 크므로
> “객체 특이적 의미 상호작용”이라고 결론

내립니다. 

그런데 이것은 논리적으로 한 단계 점프합니다.

cross-concept control에서

$$
\gamma(X,X)>\gamma(X,Y)
$$

가 관찰되었다면,

> “γ가 단순한 global modality effect만으로 설명되지 않는다”

정도는 말할 수 있습니다.

하지만 이것만으로

> “γ는 object-specific semantic interaction이다”

까지 확정하기는 어렵습니다.

예를 들어 object frequency, caption template, visual salience, image-editing characteristics, object size 등이 개념별로 다르면 그것들도 γ에 영향을 줄 수 있습니다.

따라서 여기서는

**현재 표현**

> 객체 특이적 의미 상호작용

보다는

**더 안전한 표현**

> object-conditioned cross-modal interaction

또는

> object-specific interaction consistent with semantic grounding

정도가 적절합니다.

특히 “semantic”은 상당히 강한 단어입니다.

---

# 2-2. cross-concept 0.67%를 “귀무”라고 부르는 것은 잘못입니다

이 부분은 현재 버전에서 새롭게 생긴 가장 명확한 논리적 문제입니다.

논문은:

> 다른 개념 X의 이미지와 Y의 텍스트를 결합했더니 0.67%가 나왔다. 이것이 이 데이터의 귀무이며, 주효과 지배가 순서를 결정할 때 예측되는 값이다.

라고 합니다. 

하지만 **0.67%가 왜 null expectation인지 수학적으로 도출되어 있지 않습니다.**

이건 단순히 empirical negative control입니다.

더욱이 X 이미지 + Y 텍스트에는 애초에 “정답 diagonal”이라는 의미가 없습니다. X와 Y가 서로 다른 개념이므로 원래의 2×2 semantic correspondence가 깨집니다.

따라서 0.67%는

> “negative-control accuracy”

라고 불러야지,

> “null distribution”

이라고 부르면 안 됩니다.

그리고 더 중요한 문제는:

> “주효과 지배가 순서를 결정하면 0.67%가 예상된다.”

라는 부분입니다.

그걸 보이려면 실제로 \(\alpha,\beta,\gamma\)의 null-generating mechanism을 정의하고 시뮬레이션하거나 분석적으로 유도해야 합니다.

현재 논문에는 그 derivation이 없습니다.

**수정 권고:**

> “이는 무작위 chance baseline이 아니라, 개념 간 잘못된 pairing에서 얻은 empirical negative control이다.”

라고 명시하는 것이 안전합니다.

---

# 2-3. “코사인은 귀무보다 6배다”는 문장도 별로 좋은 논리가 아닙니다

현재:

$$
4.03\% / 0.67\%\approx6
$$

이라는 점을 강조합니다. 

하지만 0.67% 자체가 통계적 null이 아니기 때문에 “6배”는 과학적으로 큰 의미가 없습니다.

오히려 reviewer가

> “Why should six times an arbitrary negative-control accuracy be meaningful?”

이라고 물을 것입니다.

이 숫자는 삭제해도 됩니다.

더 강한 결과는 이미 있습니다.

$$
\gamma>0
$$

이고

$$
\gamma<\max(|\alpha|,|\beta|)
$$

이므로 matching이 실패한다는 것입니다.

여기에 0.67%를 억지로 추가할 필요가 없습니다.

---

# 2-4. “γ는 42개 개념 전부에서 bootstrap 95% CI가 양수”는 표현을 다시 확인해야 합니다

현재 문장은:

> “γ는 42개 개념 전부에서 부트스트랩 95% CI가 엄격히 양수다(매크로 CI [0.00111, 0.00130]).”

입니다. 

그런데 이 문장은 두 가지 수준을 섞습니다.

1. **각 개념별 CI**
2. **42개 개념을 평균한 macro CI**

“42개 개념 전부에서 CI가 양수”라는 주장이라면 개념별 42개의 CI가 각각 보고되어야 합니다.

반면 뒤에 제시한

$$
[0.00111,0.00130]
$$

은 macro-level CI일 뿐입니다.

이 둘은 전혀 같은 것이 아닙니다.

따라서 실제 계산이 정말 개념별 bootstrap CI까지 했는지 확인해야 합니다.

만약 macro bootstrap만 한 것이라면 문장을 반드시:

> “The macro-level bootstrap CI excludes zero.”

로 바꿔야 합니다.

이건 사소한 문장 문제가 아니라 통계적 claim의 수준 문제입니다.

---

# 2-5. 2,480쌍 → 42개 개념 → 모델 9개로 확장되면서 분석 단위가 복잡해졌습니다

현재 결과는

> 42개 개념
> 2,480쌍
> 9개 모델
> 378개 concept-model combinations

을 동시에 사용합니다. 

그런데 통계적 독립성 문제가 생깁니다.

예를 들어

$$
378 = 42\times9
$$

이지만 378개가 독립적인 observation은 아닙니다.

같은 concept가 여러 모델에서 반복되고, 같은 이미지/텍스트 구조가 반복됩니다.

따라서

> “γ가 가장 작은 것이 378개 중 368개”

는 descriptive statistic으로는 괜찮지만,

> “97.4%의 독립적인 경우에서”

같은 해석은 하면 안 됩니다.

또한 9개 모델의 architecture/data/objective/fine-tuning variation이 정확히 어떻게 구성되었는지가 2페이지에서는 거의 보이지 않습니다.

**리뷰어 입장에서는 이 9-model 결과가 오히려 본문 공간을 많이 차지하면서도 재현성을 떨어뜨립니다.**

차라리 핵심 모델 하나를 본문에 두고 9개 모델 robustness를 작은 표 하나로 넣는 편이 낫습니다.

---

# 2-6. 가장 중요한 개념적 문제: “정보가 있다”와 “cross-modal interaction이 있다”를 너무 쉽게 연결하고 있습니다

현재 논리:

$$
\text{linear probe AUC}>chance
$$

→ 정보가 표현되어 있음

그리고

$$
\gamma>0
$$

→ cross-modal interaction이 존재함

이 둘은 각각 맞는 주장입니다.

하지만 논문 전체의 framing은 이 두 결과를 합쳐

> “information exists but is not sufficiently strong at the interface”

라고 갑니다.

여기서 조심해야 합니다.

선형 probe가 검출하는 것은

$$
w_I^\top v
$$

또는

$$
w_T^\top t
$$

라는 **특정 분류 방향에서의 decodability**입니다.

반면 γ는

$$
v^\top t
$$

의 특정 2×2 contrast입니다.

따라서

> “probe에서 정보가 검출되므로 같은 정보가 similarity에 존재하지만 약하다”

라고 해석하면 안 됩니다.

보다 정확하게는:

> “The modalities contain linearly decodable state information, while the standard cosine interface produces only a weak corresponding interaction term.”

입니다.

즉 **동일한 정보의 약한 버전**이라고 단정하지 말고, 두 현상의 공존을 보여주는 것이 안전합니다.

---

# 2-7. 26.13% probe score는 아직도 비교가 공정하지 않습니다

이 부분은 논문도 스스로

> “개념별 프로브를 쓰므로 개념 비의존 코사인과 대칭이 아니다.”

라고 인정합니다. 

그런데도 figure에서

> “같은 블록에서 6배”

라고 강조합니다.

이건 reviewer가 쉽게 공격할 수 있습니다.

왜냐하면 비교 대상은

* CLIP: 고정된 global cosine
* probe: concept-specific learned classifiers + learned decision function

이기 때문입니다.

즉 computational access와 supervision level이 다릅니다.

다행히 뒤에서 GroupKFold + global \(W\)를 사용한 실험을 추가했으므로 **26.13% 결과는 사실상 불필요합니다.**

저라면 본문에서 줄입니다.

핵심은:

$$
\text{cosine}=4.03\%
$$

대

$$
v^\top Wt=29.19\%
$$

입니다.

이 비교가 훨씬 공정합니다.

---

# 2-8. rank-32의 29.19%를 “같은 자리에 도달”이라고 해석하는 것은 과합니다

논문은

> “개념별 지식 없이도 같은 자리에 도달하므로 … 결론은 개념 지식이 아니라 채점 규칙에 관한 것이다.”

라고 합니다. 

하지만 rank-32 \(W\)는 **학습된 global transformation**입니다.

따라서 개념별 knowledge leakage는 없더라도,

> “단순한 scoring rule만 바꿨다”

라고 하기에는 애매합니다.

\(W\)가 사실상 새로운 cross-modal model을 학습한 것이기 때문입니다.

특히 rank-32면 parameter 수가 상당합니다.

물론 full \(W\)보다 훨씬 작다는 점은 장점입니다.

그러나 결론은

> “the failure can be partially alleviated by learning a low-rank bilinear scoring function on frozen embeddings”

정도가 정확합니다.

“코사인의 문제를 scoring rule만 바꿔 해결했다”는 식으로 일반화하면 안 됩니다.

---

# 2-9. “필요한 것은 rank 1”이라는 문장은 현재 논문의 논리와 충돌합니다

3.3에서

> “반대쪽 극단인 완전 W(262,144개 파라미터)는 rank 32보다 낮은 23.91%”

라고 하면서 rank-32가 좋다고 합니다. 

그런데 결론에서는

> “필요한 것은 옳은 저계수 구조이고”

라고 합니다.

이 정도는 괜찮습니다.

하지만 이전 버전의 문구였던

> “필요한 것은 rank 1이다.”

같은 주장은 현재 결과로는 뒷받침되지 않습니다.

오히려 현재 결과는:

* rank 1도 상당히 좋아짐
* rank 2~32가 더 좋아질 가능성
* rank 32가 현재 best
* rank 32와 2 사이 CI가 겹침
* optimum rank는 모름

입니다.

실제로 한계에서도 이를 인정합니다. 

따라서 “필요한 것은 rank 1”이라는 표현은 완전히 제거하는 게 맞습니다.

---

# 2-10. geometry 분석은 흥미롭지만 현재 논리에서 가장 취약한 부분 중 하나입니다

현재:

$$
\gamma \approx \frac14\|d_I\|\|d_T\|\cos(d_I,d_T)
$$

그리고

$$
\cos(d_I,d_T)=0.167
$$

$$
\rho(\gamma,\cos(d_I,d_T))=0.742
$$

를 보여줍니다. 

이건 상당히 흥미로운 분석입니다.

하지만 바로 다음에

> “교차항이 작은 것은 두 모달리티의 극성 방향이 서로 정렬되어 있지 않기 때문이다.”

라고 결론내리는 것은 조금 과합니다.

왜냐하면

$$
\gamma
\propto
\|d_I\|\|d_T\|\cos(d_I,d_T)
$$

이므로 γ의 크기는 세 요인에 의해 결정됩니다.

1. image difference magnitude
2. text difference magnitude
3. direction alignment

그런데 correlation이 높다는 것만으로

> alignment가 causal bottleneck이다

라고 할 수 없습니다.

정확한 표현은:

> “The small interaction is consistent with weak alignment between image and text state-difference directions.”

정도가 좋습니다.

---

# 3. 새로 생긴 가장 중요한 문제: 42개 개념으로 늘렸는데 왜 9개 모델 결과가 갑자기 필요한가?

이건 논문 설계 측면의 문제입니다.

현재 논문은 상당히 많은 것을 넣었습니다.

* 42 concepts
* 2,480 pairs
* image probe
* text probe
* positional control
* 2×2 decomposition
* negative control
* bootstrap
* 9 models
* rank-1~32 W
* random W
* shuffled labels
* full W
* geometry
* T2I retrieval

2페이지에 이걸 다 넣으면 각각의 실험이 **무엇을 증명하는지 독자가 놓칩니다.**

현재 가장 중요한 causal chain은 사실 4개입니다.

$$
\boxed{\text{Probe}}
$$

→ 정보는 decodable

$$
\boxed{\text{2×2}}
$$

→ interaction은 존재하지만 작음

$$
\boxed{\text{W}}
$$

→ scoring interface를 바꾸면 개선 가능

$$
\boxed{\text{T2I}}
$$

→ 이 decomposition이 실제 downstream behavior와도 연결됨

이 네 개만 명확하게 하면 됩니다.

현재는 실험이 많아져서 오히려 중심 주장이 흐려질 위험이 있습니다.

---

# 4. T2I 실험은 이번 버전에서 상당히 좋아졌지만, 해석을 더 좁혀야 합니다

현재:

> signed β predicts the R@1 drop with \(r=0.835\), \(p=0.005\)

라고 합니다. 

이건 좋은 결과입니다.

특히 단순히 “negation accuracy가 낮다”가 아니라

> decomposition에서 예측한 β가 실제 downstream degradation과 연결된다

는 점은 논문의 실용적 의미를 높입니다.

하지만 여기서도 **9개 모델뿐이라는 점**이 상당히 큽니다.

$$
n=9
$$

에서

$$
r=0.835
$$

는 흥미롭지만 robust statistical evidence라고 하기 어렵습니다.

p=0.005가 나오더라도 n이 너무 작기 때문에 effect estimate 자체가 불안정합니다.

따라서

> “strong evidence”

보다는

> “consistent cross-model association”

정도로 쓰는 것이 좋습니다.

그리고 가능하면 모델 수를 늘리는 것보다 **bootstrap CI for r**를 보여주는 게 더 낫습니다.

---

# 5. 제목은 이전보다 좋아졌지만 아직 약간 애매합니다

현재 제목:

> Object-Presence Negation in Multi-Object Scenes: Represented Information versus Similarity Interaction

은 방향은 좋습니다.

하지만 “represented information”이 무엇인지 제목만으로는 모호합니다.

논문의 핵심은 사실:

> **Decodability ≠ similarity interaction**

입니다.

따라서 오히려 이런 식의 제목이 논문의 novelty를 더 직접적으로 전달합니다.

> **When Representation Is Not Enough: Weak Cross-Modal Interaction for Negation in CLIP**

혹은 더 중립적으로

> **Decodable but Not Decisive: Cross-Modal Interaction Limits Negation in CLIP**

현재 제목도 나쁘지는 않지만, 리뷰어 입장에서 첫 번째 제목이 논문의 주장과 더 직접적으로 연결됩니다.

---

# 6. 현재 버전에서 가장 좋은 결과와 가장 약한 결과를 구분해야 합니다

### 가장 강한 결과

저는 이 네 가지를 남기겠습니다.

**① AB-swap text probe**

$$
64.69\%
$$

→ 단순 negation-token shortcut만으로 설명되지 않음. 

**② interaction magnitude**

$$
|\alpha|=0.00380,\quad
|\beta|=0.00628,\quad
\gamma=0.00121
$$

→ γ가 존재하지만 dominant main effect보다 5.2배 작음. 

**③ 2×2 identity**

$$
\Delta>0
\iff
\gamma>\max(|\alpha|,|\beta|)
$$

→ empirical ranking을 정확히 설명. 

**④ global low-rank \(W\)**

$$
4.03\%\rightarrow29.19\%
$$

→ frozen embedding에서도 scoring interface 변경으로 상당한 개선 가능. 

이 네 개만으로도 논문은 충분히 서사가 생깁니다.

---

# 7. 반대로 삭제하거나 크게 축소해도 되는 부분

저라면 다음은 줄입니다.

### 26.13% probe-based scoring

비교가 unfair합니다.

### random rank-1 0.44%, shuffled 0.69%

흥미롭지만 핵심 논지를 강화하는 정도가 작습니다.

### full W 23.91%

rank-32가 더 좋다는 것은 흥미롭지만, full W가 overfit되었다는 것 이상의 해석은 어렵습니다.

### “7.2배”

ratio 자체는 인상적이지만 absolute performance가 29.19%밖에 안 된다는 점을 함께 보면 과도하게 강조하면 안 됩니다.

### T2I

좋은 auxiliary experiment이지만 9 models라면 main claim의 중심에 놓지 않는 게 좋습니다.

---

# 8. 가장 중요한 수정 우선순위

제가 저자라면 다음 순서로 고칩니다.

**1순위 — cross-concept 0.67%의 해석 수정**

“null” → “empirical negative control”

그리고 “주효과 지배에서 0.67%가 예측된다”는 문장은 삭제하거나 별도의 simulation/derivation을 추가합니다.

**2순위 — 42개 개념 CI 표현 정리**

“42개 개념 전부에서 CI가 양수”와 “macro CI가 양수”를 엄격히 구분해야 합니다.

**3순위 — probe와 γ의 관계를 약하게 표현**

“같은 정보가 약하게 표현된다”가 아니라

> “decodable state information coexists with weak interaction at the cosine interface”

라고 해야 합니다.

**4순위 — rank-32 \(W\)의 claim 축소**

“scoring rule만 바꾸면 해결된다”가 아니라

> “a learned low-rank bilinear scoring function can substantially recover the lost matching signal”

정도로.

**5순위 — geometry causal language 완화**

“alignment가 원인이다” → “weak alignment is consistent with / associated with weak γ”.

**6순위 — 2페이지에서 실험 수 줄이기**

지금은 너무 많이 넣었습니다. 실험을 추가하는 단계가 아니라 **버리는 단계**입니다.

---

# 9. 이전 버전과 비교하면

간단히 평가하면 이렇습니다.

| 문제                       | 이전     | 수정본                              |
| ------------------------ | ------ | -------------------------------- |
| decomposition novelty 과장 | 심각     | **개선됨**                          |
| 기존 연구와의 충돌               | 상당함    | **크게 개선**                        |
| alignment upper bound    | 논리적 문제 | **대폭 개선**                        |
| positional shortcut      | 취약     | **좋아짐**                          |
| chance baseline          | 취약     | **개선됐지만 0.67% 해석 오류**            |
| artifact 문제              | 존재     | **여전히 존재**                       |
| probe vs cosine 공정성      | 문제     | **W 실험으로 개선됐지만 26.13%는 여전히 불공정** |
| γ의 의미                    | 과장 가능  | **대체로 개선**                       |
| cross-modal geometry     | 없음/약함  | **흥미로운 신규 분석, 단 causal 해석은 과함**  |
| downstream validation    | 약함     | **T2I로 개선**                      |
| 전체 논리                    | 다소 산만  | **훨씬 명확해짐**                      |
| 주요 약점                    | 이론적 과장 | **통계적 해석과 claim calibration**    |

---

## 최종 판정

현재 버전은 **이전 버전보다 확실히 논문에 가까워졌습니다.**

특히 가장 중요한 변화는

> **“CLIP에 negation 정보가 없다”**

에서

> **“negation 관련 정보는 decodable하지만, 그것이 standard cosine similarity에서 충분히 강한 cross-modal interaction으로 나타나지 않는다.”**

로 논문의 질문을 바꾼 것입니다.

이 방향은 유지하는 것이 맞습니다.

다만 현재 가장 위험한 부분은 더 이상 수학이 아닙니다. **통계적 결과에 붙인 해석이 데이터보다 한 발씩 앞서 있습니다.**

특히

1. 0.67%를 null이라고 부르는 것,
2. macro CI를 개념별 CI처럼 읽힐 수 있게 쓰는 것,
3. probe information과 γ를 동일한 정보로 간주하는 것,
4. rank-32 \(W\)를 단순 scoring-rule change로 과도하게 해석하는 것,
5. \(\cos(d_I,d_T)\)와 γ의 상관을 bottleneck의 원인으로 해석하는 것

이 다섯 가지는 수정해야 합니다.

그리고 **새로운 실험을 더 추가하는 것보다 현재 실험의 해석을 엄밀하게 만드는 것이 우선**입니다. 현재 결과만으로도 핵심 주장은 충분히 만들 수 있습니다. 특히 논문의 가장 방어하기 좋은 문장은 결국 이것입니다.

> **Negation-related state information is linearly decodable from both modalities, yet its corresponding cross-modal interaction is consistently too weak relative to modality-specific main effects to determine cosine-similarity rankings.**

이 수준으로 claim을 제한하면, 현재 논문은 “2×2 decomposition을 한 논문”이 아니라 **representation-level decodability와 decision-level usability 사이의 정량적 gap을 측정한 논문**으로 읽힐 수 있습니다.

### 논문 개요 및 개정 사항 요약 (Overview & Revision Assessment)

개정된 판본(`PAPER_2PAGE_2.pdf`)은 이전 버전(`PAPER_2PAGE.pdf`)에서 제기되었던 주요 학술적 질문들을 적극적으로 수용하여, 논문의 정체성을 '완전한 통제 조건 하에서의 기전 진단(Mechanistic Diagnosis)'으로 재정의하고 실증적 완성도를 크게 끌어올렸습니다.

```
[이전 판본(v1) 대비 주요 변경 사항]
1. 제목 및 스코프 구체화: "CLIP의 부정 매칭 실패..." -> "다객체 장면의 객체 존재 부정..."으로 변경하여 프레임의 적용 범위를 명확히 한정[cite: 1, 3].
2. 일반화 가능성 검증 (3.3절): 개념별 프로브 외적(24.96%)에 머무르지 않고, 미지 개념(Unseen concepts)에 대한 GroupKFold 교차 검증 하에서 단일 공유 저계수 행렬(Rank-32 W, 29.19%)의 유효성을 실증[cite: 1, 3].
3. 대조군(Control) 설계 강화 (3.2 & 3.3절): 무작위 외적(0.44%), 라벨 셔플(0.69%), 완전 변환 W(23.91%) 및 결합 없는 실측 귀무선(0.67%)을 추가하여 성능 향상이 '파라미터 수'가 아닌 '저계수 구조(Inductive Bias)'에서 기인함을 증명[cite: 3].
4. 외부 과제 예측 타당성 정교화 (3.4절): T2I 갤러리 검색에서 부정으로 인한 성능 하락폭(Delta R@1)을 부호 있는 beta가 r = +0.835 (p = 0.005)로 예측함을 명시[cite: 3].

```

---

### 주요 강점 (Key Strengths)

* **Winoground Group Score 실패에 대한 완결적 대수적 설명:**
Winoground(Thrush et al., 2022)가 제기한 '무작위 확률(16.67%) 이하의 모델 붕괴' 현상을 $S_{ab} = C + a\beta + b\alpha + ab\gamma$ 분해와 $\Delta = 2\gamma - 2\max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$ 항등식을 통해 완벽히 규명했습니다. 본 논문은 모델이 실패하는 이유가 결합 신호의 완전한 부재($\gamma = 0$)가 아니라, $\gamma(0.00121)$가 주효과 $\max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})(0.00628)$의 1/5.2 수준에 불과하여 결정론적 역전이 강제되기 때문임을 실측 데이터로 명쾌하게 밝혔습니다.


* **'개념 의존성' 비판에 대한 성공적인 실증 방어 (3.3절):**
이전 버전의 선형 프로브 법선 외적($w_I w_T^T$)이 "개념별 지도 학습 정보를 사전에 요구한다"는 한계를 극복하기 위해, 미지 개념 홀드아웃(GroupKFold) 조건에서 $S_W = v^T W t$를 평가했습니다. 공유 저계수 변환(Rank-32)이 학습 데이터에 없던 새로운 개념에서도 29.19%(코사인 4.03% 대비 7.2배)에 도달함을 보임으로써, 병목이 '개념 지식'이 아닌 '항등 행렬로 고정된 코사인 유사도 인터페이스' 자체에 있음을 설득력 있게 입증했습니다.


* **체계적인 대조군(Ablation & Control) 설계를 통한 인과성 확립:**
* **실측 귀무 가설(0.67%):** 개념 간 캡션을 무작위 교차 결합한 실측 귀무선(0.67%)을 제시하여, 코사인의 4.03%가 완전한 무정보가 아니라 '미약하지만 유의미한 교차항 $\gamma$'을 반영하고 있음을 보였습니다.


* **파라미터 수 vs 구조:** 무작위 벡터(0.44%), 셔플 프로브(0.69%), 완전 $W$(23.91%)와의 비교를 통해 29.19%의 이득이 단순 과적합이나 파라미터 용량 증가가 아닌, 올바른 저계수 극성 정렬 구조에서 비롯됨을 확인했습니다.




* **선행 연구 간 가설 충돌의 이론적 통합:**
Alshehri et al.(2026, LCSE)의 "유사도가 개념 증거를 평균 풀링한다", Koishigarina et al.(2026)의 "결합 정보는 인코더 내부에 있다", Kang et al.(2025, DCSM)의 "공유 벡터 공간의 기하학적 모순", Lu et al.(2026, PeakPatch)의 "최종 레이어 표상 붕괴" 등의 논의를 단일 4계수 좌표계($C, \alpha, \beta, \gamma$) 안에서 상호 보완적인 현상으로 명쾌하게 정리했습니다.



---

### 비판적 검토 및 잔여 한계점 (Weaknesses & Critical Inquiries)

* **미지 개념 Rank-32 변환의 일반화 경계 및 복잡도:**
GroupKFold에서 Rank-32 행렬 $W$가 29.19%를 달성한 것은 고무적이나, 여전히 $2 \times 2$ 상한선(100%) 대비 상당한 격차가 존재합니다.


* 왜 Rank-1(단일 극성 축 외적)에서 Rank-32로 확장될 때 성능이 점진적으로 증가하는지, 다차원 잠재 공간에서 추가된 31개의 랭크가 어떠한 시각-언어적 불일치(Modality gap 또는 기하학적 뒤틀림)를 보정하는지에 대한 기하학적 설명이 보강되어야 합니다.
* COCO 42개 개념 외에 완전히 다른 도메인(예: PUG:SPAR, CLEVR, NegBench Open-domain)에서도 동일한 $W$가 전이(Transfer)되는지 추가 검증이 필요합니다.




* **부정 형태의 제한성 및 단일 객체 부정과의 단절:**
저자들이 한계(Limitations)에서 정직하게 인정했듯이, 본 프레임워크는 $a, b \in {+1, -1}$을 정의할 수 있는 다객체 AB-swap 구조에 강하게 의존합니다.


* "a photo with no dogs"와 같은 단일 객체 부재 질의의 경우, 텍스트의 반대 상태 $b$를 구성하기 위한 참조 앵커가 모호해져 분해식 적용이 까다로워집니다.


* 속성 부정(e.g., "red ball without stripes")이나 행위 부정(e.g., "dog not running")으로 분해 프레임워크를 확장하기 위한 수식적 정의($2^k$ 요인 설계 등)가 제시된다면 연구의 파급력이 더욱 커질 것입니다.




* **외부 과제 지표 해석의 비대칭성 (3.4절):**
부호 있는 $\beta$가 COCO T2I 검색의 '성능 하락폭(Delta R@1)'을 강하게 예측($r = +0.835$)하지만, 부정 질의 자체의 절대 성능이나 $\gamma/\beta$ 비율은 예측하지 못함($r = -0.289$)을 기술하고 있습니다.


* 이는 본 프레임워크가 "부정어가 시스템에 가하는 간섭/왜곡의 크기"를 진단하는 데는 탁월하지만, 모델의 절대적 검색 역량을 직접 평가하는 단일 지표로 쓰이기에는 보조 지표가 추가로 필요함을 시사합니다. 이 비대칭성에 대한 이론적 고찰을 논의 섹션에 추가할 필요가 있습니다.



---

### 완결 논문(`PAPER.md`) 확장을 위한 제언 (Recommendations for Full Paper)

1. **테스트 타임 개입(TTM) 및 벤치마크 착시 분석 심화:**
GroupMatch($S_{11} + S_{22} > S_{12} + S_{21}$) 계열의 기법들이 주효과 $\alpha, \beta$를 대수적으로 강제 상쇄하여 $\gamma > 0$만을 평가하는 벤치마크 편법임을 지적하고, 실제 1:N 검색에서는 $\beta$가 소거되지 않아 반드시 $\gamma > \vert{}\beta\vert{}$가 요구된다는 점을 실증 검색 데이터와 결합하여 명시하십시오.


2. **부정 미세조정(NegCLIP, ConCLIP)의 대수적 병리 수록:**
Sammani et al.(2026)에서 관찰된 ConCLIP의 표상 붕괴(Affirmation bias의 과도한 음수 역전)를 본 논문의 계수 분포 표/그래프로 시각화하여, "미세조정 모델들이 $\gamma$를 키우는 대신 $\alpha$를 음수로 밀어내어 긍정 정확도를 파괴한다"는 통찰을 부각하면 완성도가 극대화될 것입니다.


3. **중간 레이어 표상(PeakPatch) 및 토큰 맵(DCSM)과의 접점 논의:**
DCSM(Kang et al., 2025)이나 PeakPatch(Lu et al., 2026)처럼 토큰/패치 레벨 또는 중간 레이어에서 극성을 복원하는 접근법들이, 결과적으로 출력단에서 $\gamma$를 키우는 효과를 내는지 아니면 $\beta$를 억제하는 효과를 내는지를 본 분해식으로 비교 분석하는 절을 추가할 것을 권장합니다.



---

### 심사 결과 요약 (Verdict)

* **판정:** **Strong Accept (최우수 진단 논문)**
* **총평:** 개정판은 2페이지 분량의 극단적인 공간 제약 속에서도 수식적 엄밀함, 방법론적 통제, 그리고 일반화 가능한 저계수 $W$의 실증까지 압축적으로 담아낸 매우 훌륭한 연구입니다. 불필요한 과장을 배제하고 과학적으로 정직하게 한계와 성능을 기술한 태도가 돋보입니다. 제기된 잔여 질문들을 완결 논문(`PAPER.md`)에 보강한다면 주요 최우수 국제 인공지능 학술대회(NeurIPS / ICLR / CVPR) 메인 트랙에 충분히 게재될 수 있을 것으로 평가합니다.

### 종합 평가 및 판정 (Overall Recommendation)

**최종 판정: Reject (거절)**

본 논문(PAPER_2PAGE_2.pdf)은 이전 판본(PAPER_2PAGE.pdf)에 비해 연구의 프레임을 '다객체 장면의 객체 존재 부정'으로 명확히 한정하고, 미지 개념(Unseen concepts)에 대한 공유 저계수 변환(Rank-32 $W$) 실험 및 다양한 대조군(Control baselines)을 보강하여 분석의 엄밀성을 크게 개선하였습니다.

그러나 학술 심사에서 "엄격한 기준을 적용하여 검증된 연구만을 채택한다"는 원칙에 입각하여 평가할 때, 본 논문은 여전히 **(1) 지나치게 인공적인 $2 \times 2$ 대칭 AB-swap 프레임워크의 구조적 한계, (2) 제안된 대안(Rank-32 Bilinear)의 낮은 절대 성능 상한선(29.19%), (3) 외부 과제에서 절대 검색 성능을 예측하지 못하는 진단적 결함($r = -0.289$), (4) 인페인팅 아티팩트의 주효과 오염 가능성**이라는 치명적인 약점을 안고 있어 현 형태로는 게재 승인이 어렵습니다.

---

### 논문 요약 및 이전 판본 대비 변경점 (Summary & Revisions from v1)

개정된 논문은 CLIP의 부정 매칭 실패 원인을 $2 \times 2$ 요인 분해(Factor Decomposition)를 통해 규명하고자 합니다. 유사도 점수를 $S_{ab} = C + a\beta + b\alpha + ab\gamma$ ($a, b \in \{+1, -1\}$)로 분해하여 부정 검색의 성공 조건이 $\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$임을 보이고, 이를 ViT-B/32 모델 및 42개 개념의 2,480개 최소쌍에서 실측합니다.

이전 판본(v1)과 비교하여 다음과 같은 점들이 수정·보완되었습니다:

1. **연구 범위의 명확화:** 논문 제목과 범위를 일반 부정 전체가 아닌 "다객체 장면의 객체 존재 부정"으로 수정하여 한계를 명확히 설정하였습니다.


2. **미지 개념 일반화 검증 추가 (3.3절):** 개념별 프로브 외적(26.13%)에 머무르지 않고, 미지 개념에 대한 GroupKFold 교차 검증 하에서 단일 공유 저계수 행렬(Rank-32 $W$, 29.19%)을 평가하여 채점 규칙의 일반화 가능성을 확인하였습니다.


3. **대조군(Control Baselines) 확충 (3.2 & 3.3절):** 무작위 외적(0.44%), 라벨 셔플(0.69%), 완전 $W$(23.91%) 및 결합 없는 실측 귀무선(0.67%)을 제시하여 이득이 단순 파라미터 수가 아닌 저계수 구조적 편향(Inductive bias)에서 기인함을 입증하였습니다.


4. **외부 과제 예측성 정교화 (3.4절):** COCO 5,000장 갤러리 T2I 검색에서 부정으로 인한 R@1 하락폭을 부호 있는 $\beta$가 $r = +0.835$ ($p = 0.005$)로 예측함을 명시하였습니다.



---

### 주요 강점 (Strengths)

* **Winoground Group Score 실패의 대수적 정식화:** Winoground에서 관찰되는 무작위 확률(16.67%) 이하의 모델 붕괴 현상을 $S_{ab} = C + a\beta + b\alpha + ab\gamma$ 분해 및 $\Delta = 2\gamma - 2\max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$ 항등식을 통해 직관적이고 완결성 있게 증명하였습니다. 실패 원인이 신호 부재($\gamma = 0$)가 아니라 $\gamma(0.00121)$ 대비 지배 주효과 $\max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})(0.00628)$의 불균형(1/5.2) 때문임을 실측 데이터로 명확히 밝혔습니다.


* **철저한 대조군 실험:** 실측 귀무선(0.67%), 무작위 외적(0.44%), 셔플 프로브(0.69%), 26만 개 파라미터의 Full $W$(23.91%) 등 다각도의 통제군을 배치하여 저계수 극성 정렬 구조의 필요성을 효과적으로 뒷받침하였습니다.


* **선행 연구의 이론적 통합:** Alshehri et al. (LCSE)의 '개념 증거 평균 풀링' 주장과 Koishigarina et al. (LABCLIP)의 '인코더 내부 결합 정보 존재' 주장을 단일 4계수 좌표계($C, \alpha, \beta, \gamma$) 내에서 상호 모순 없는 현상으로 성공적으로 조율하였습니다.



---

### 주요 약점 및 거절 사유 (Weaknesses & Reasons for Rejection)

**1. 대칭적 AB-Swap에 종속된 $2 \times 2$ 프레임워크의 실용적 한계**

* 본 연구의 요인 분해식은 텍스트 상태 $b \in {+1, -1}$를 엄격히 정의하기 위해 긍정과 부정이 대칭을 이루는 인공적 AB-swap 문장("A but no B" vs "B but no A")을 필수 전제로 요구합니다.


* 그러나 실제 검색 환경과 NegBench, VALSE, SimpleNeg 등 표준 벤치마크에서 다루는 대부분의 부정 질의는 "a photo of a room with no chairs", "umbrella and no person"과 같은 비대칭적 단일/복합 부정입니다.


* 저자 스스로 한계에서 단일 객체 부정을 다루지 못한다고 기술하였으나, 이는 본 연구의 진단 도구가 실제 응용 및 광범위한 부정 질의 평가에 직접 적용될 수 없음을 의미하며 기여의 범위를 크게 제약합니다.



**2. 대안 모델(Rank-32 Bilinear)의 낮은 절대 성능 및 SOTA 사후 보정 기법과의 격차**

* 저자들은 항등 행렬로 고정된 코사인 유사도(4.03%) 대신 제안한 Rank-32 공유 변환이 29.19%로 7.2배 향상되었다고 강조합니다.


* 그러나 $2 \times 2$ 판정의 4지선다 우연 확률이 16.67%임을 감안할 때, 29.19%는 여전히 전체 쌍의 70% 이상을 틀리는 매우 낮은 성능입니다.


* 반면, 동결된 CLIP을 활용하는 최근의 훈련 불필요(Training-free) 및 경량 사후 적응 연구들(예: LCSE, SpaceVLM-DRC, PeakPatch, NEGTOME)은 NegBench MCQ나 FACTOR-Bench 등에서 60~85% 이상의 높은 정답률을 달성하고 있습니다. 제안된 저계수 헤드가 단순 진단용 프로브를 넘어 실질적 대안으로 기능하기에는 성능적 설득력이 부족합니다.



**3. 외부 과제 진단에서 절대 검색 성능 예측 실패 (3.4절)**

* 3.4절에서 부호 있는 $\beta$가 COCO T2I 검색의 '성능 하락폭(Delta R@1)'을 $r = +0.835$로 유의미하게 예측한다고 보고하였으나, 정작 부정 질의의 절대 검색 성능이나 핵심 진단 비율인 $\gamma/\beta$는 검색 성능을 전혀 예측하지 못함($r = -0.289$)을 밝혔습니다.


* 이는 본 분해 프레임워크가 "부정이 모델에 미치는 상대적 교란 크기"는 추적할 수 있지만, "모델이 부정 질의를 올바르게 수행하는지"에 대한 절대적 역량은 설명하지 못한다는 치명적인 진단적 결함을 드러냅니다.



**4. 인페인팅(BEAF) 아티팩트에 의한 이미지 주효과($\beta$) 오염 가능성**

* 이미지 최소쌍을 생성하기 위해 BEAF의 인페인팅 제거 이미지를 사용하였습니다.


* 인페인팅 과정에서 발생하는 미세한 블러, 텍스처 불연속성, 경계선 잔상 등 생성 아티팩트가 CLIP 비전 인코더의 임베딩에 영향을 미쳐 이미지 주효과 $\beta$의 크기($\vert{}\beta\vert{} = 0.00628$)를 인위적으로 부풀렸을 가능성이 존재합니다.


* 자연 상태에서 객체가 원래 없는 실제 이미지와의 교차 검증이 배제되어 있어, 주효과 지배 현상이 모델의 내재적 특성인지 데이터 생성 아티팩트의 산물인지 완전히 분리되지 않았습니다.



**5. 2페이지 판본 내 다중 모델 검증 근거 부족**

* 서론 및 3.2절에서 아키텍처 3종, 학습 데이터 3종, 손실함수 2종, 부정 미세조정 4종 등 9개 모델에 대한 전수 조사를 언급하고 있으나, 2페이지 본문에 수록된 수치와 그래프는 ViT-B/32 단일 모델에 대한 결과뿐입니다.


* SigLIP, NegCLIP, ConCLIP 등 다양한 목적함수와 아키텍처에서 $\gamma \ll \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$ 규칙이 보편적으로 성립하는지 확인할 수 있는 정량적 표가 논문 본문에 누락되어 있어 주장의 완전한 검증이 불가능합니다.



---

### 저자 대상 질의 (Questions for Authors)

1. **단일 부정 확장 가능성:** $b \in {+1, -1}$와 같은 대칭적 대조 문장이 성립하지 않는 단일 객체 부재 질의("a street with no cars")에 대해 본 요인 분해식을 수학적으로 어떻게 일반화할 수 있습니까?


2. **Rank-32의 기하학적 의미:** Rank-1(단일 극성 축 외적, 24.96%)에서 Rank-32(29.19%)로 랭크를 확장할 때 성능이 점진적으로 향상되는 이유가 무엇입니까? 추가된 31개의 기저 벡터가 모달리티 갭(Modality gap)이나 시각-언어 불일치의 구체적으로 어떤 기하학적 요소를 보정하는 것입니까?


3. **인페인팅 노이즈 통제:** BEAF 인페인팅 데이터셋에서 인페인팅 영역의 아티팩트가 이미지 주효과 $\beta$에 미치는 영향을 통제하기 위해 어떤 실험적/통계적 검증을 수행하였습니까?


4. **절대 성능 예측 실패에 대한 해석:** $2 \times 2$ 최소쌍에서의 $\gamma/\beta$ 비율이 실제 COCO 5,000장 T2I 검색 성능과 상관관계가 없는($r = -0.289$) 대수적·기하학적 원인은 무엇입니까?



---

### 최종 결론 (Conclusion)

본 논문은 $2 \times 2$ 요인 분해를 통해 CLIP의 부정 매칭 실패 메커니즘을 규명하고, 개념 비의존적 저계수 변환의 가능성을 실증한 우수한 이론적·진단적 시도입니다. 그러나 $2 \times 2$ AB-swap이라는 인공적 환경에 국한된 분석 범위, 실용적 수준에 도달하지 못한 대안 성능, 절대 검색 성능과의 괴리 등으로 인해 독립된 논문으로서의 완결성이 부족합니다.

따라서 완결 논문(PAPER.md)에 예고된 광범위한 다중 모델 전수 결과, 단일 객체 및 자연 이미지로의 프레임워크 확장, 그리고 $1:N$ 실제 검색 성능과의 대수적 연결고리를 완전히 수록하여 정규 학회 풀페이퍼(Full-length paper)로 재제출할 것을 권고하며 **Reject**를 부여합니다.

# 총평

이전 버전보다 **명확하게 개선되었습니다**. 특히 제목을 객체 존재 부정으로 좁혔고, 분해 자체가 기여가 아니라는 점을 명시했으며, 보고 단위·귀무 기준선·개념 비의존 평가·외부 과제를 추가했습니다. 제가 이전에 제기한 주요 우려 중 상당 부분에 답했기 때문에, 현재 버전은 단순한 아이디어 페이퍼가 아니라 **검증 가능한 진단 프레임워크를 제시하는 논문**에 가까워졌습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/83eeeca3-4d56-4fb7-9eaf-c45ec006e6c1/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE3ZGD6L3K&Signature=1j2QPBoxysCfu0bZxRiLaD8bTQg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFpDcHpJy1SxBA6ZwKtyz7OQpbE1LAtj4DRgn7%2BAe9gNAiEAyQo5hfPXk5ADCgHhILXAoWvfZevUzxM8LQQ%2FqkTwE5Mq%2FAQImP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDPXYdCmLDDYNxTu8iCrQBJRw0KBp8h0UC28OSXIZxLEIe3I9Ri4QU9I4crTsHfSjQJKMGEuATyZJLgqzmW1uZNNFzpsGAZm9z2rCLWETZKBhgN12wZzxhn2uaz1No%2Fidx%2FtISJKynNyyYHdKsxkSi00o0p%2BiSxLMoeEI5mRYOyRySj%2FjkG7Z%2F7LBiL5wdrfZzxrtEq3vVzxbqcnbCm5jlPjRLmXpAfZQNLUuVY6GmvrheLVZTRlLnG7WEZAJBwgZLBDST%2FJe0D1yEhpJMaZtyu4GOL1AmchDmwgkZ%2Bi76iWs6Ou8XzaJb1YquJI4Kamf5fpRK78RBLx%2FUOs3GBQeWhu31Ivfqexad2IMEBPjvSkN4fahiyWRKyRgkRB94TuGt%2BhQjVSmfnGtnAysDFyERYR4jJn0gV8IjxF4K21UKSpPM59g3pAfVim5jVT0MXDItWihuosjvMb9tc75UsYA7plIGGOTgQBWvYsNZMbFyWHNjg5Yt7O7T2INCRFSkyDvd%2BpyX%2FIS0ScI%2BNX9mQcU47moXxgQpU10ViWwAhhcF8d4mn6a6sTPq9lcGSa%2BxGlBr1Mxb0hgUmSCSjN1Gas5sT5IV%2FNigBsPuaV%2BEHY33sUhxZTbcAW%2F48z6W%2FQOn1KvEigv%2Fu6NP3ucjoQotMNRRrEhuhTeoyPFfUhUNFdXjBj316d7wqklAxKC7zMokiNiwjtRbsZxr6E1CVF36PaWZ2YWm54J8agPWDlARu5aVxs3YdDiHLc5jjByHxsXv57JD1et5HYly%2FFMjwQrcWYN7Kh%2FXcdNIvaXSeO6AiD8a98wsaDW1AY6mAFT3bh2km6C12HqjfNmqXkDIvcOWHeKz6fhzk47Xp7VbRmxmd0X0HvWks3O3UhNW5T6%2B6GeJvYSVo2aRULp0ZE%2BCs9%2F%2F08shfGfYXdzRUN0gryDu0IDlEoiEncfY5SwyQSkKGRpu86RUy%2BBenrLvTmyVSZuifGuqbptf4oGMpxQwCZrsDUEN9ZfwXvnO7dDFRB4T5DSkLoHcw%3D%3D&Expires=1788190212)

다만 심사 관점에서는 여전히 **Weak Accept와 Weak Reject의 경계**에 있습니다. 핵심 아이디어와 실험 결과는 설득력이 있지만, 현재 2페이지 본문만으로는 재현성과 통계적 타당성, 특히 rank-2 이상 bilinear 변환의 학습 절차와 외부 과제의 검증 구조가 충분히 설명되지 않습니다.

**권고 판정: Weak Accept / Minor-to-Major Revision**  
상위권 비전·머신러닝 학회 기준으로는 아직 보수적으로 **Weak Reject**, 국내 학술대회나 짧은 논문 트랙이라면 **Accept 가능성**이 있습니다.

## 1. 이전 버전보다 좋아진 점

### 1.1 기여의 범위를 정확히 조정했다

이전 버전에서는 2×2 분해가 논문의 이론적 기여처럼 읽힐 여지가 있었습니다. 새 버전은 다음과 같이 이를 명확히 제한합니다.

> “분해 자체는 2×2 요인 설계의 표준 도구이며 기여가 아니다. 기여는 그 도구로 얻은 실측이다.”

이는 매우 좋은 수정입니다. 논문의 독창성이 어디에 있는지 명확해졌습니다. 이제 주장의 중심은 수식의 새로움이 아니라 다음 세 가지 실증 결과입니다.

- 교차항 \(\gamma\)가 실제로 양수이지만 주효과보다 작다는 점.
- 같은 임베딩을 다른 채점 규칙으로 읽으면 성능이 크게 향상된다는 점.
- 이 현상이 개념 비의존적인 bilinear 변환과 외부 검색 과제에서도 관찰된다는 점.

### 1.2 제목과 결론의 범위를 좁혔다

새 제목인 **“다객체 장면의 객체 존재 부정”**은 이전 제목보다 훨씬 정확하다. 관계 부정, 행위 부정, 단일 객체 부정까지 포함하는 듯한 과도한 일반화를 줄였다.

또한 한계에서 다음을 명시한 점도 긍정적이다.

- 관계 부정과 행위 부정은 다루지 않음.
- 결합할 상대가 없는 단일 객체 부정은 프레임 밖임.
- 인페인팅 기반 반사실 이미지와 자연 이미지 부재를 비교하지 않음.
- 최적 rank는 아직 판정하지 못함.

이러한 한계 공개는 결과의 신뢰도를 오히려 높인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/83eeeca3-4d56-4fb7-9eaf-c45ec006e6c1/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE3ZGD6L3K&Signature=1j2QPBoxysCfu0bZxRiLaD8bTQg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFpDcHpJy1SxBA6ZwKtyz7OQpbE1LAtj4DRgn7%2BAe9gNAiEAyQo5hfPXk5ADCgHhILXAoWvfZevUzxM8LQQ%2FqkTwE5Mq%2FAQImP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDPXYdCmLDDYNxTu8iCrQBJRw0KBp8h0UC28OSXIZxLEIe3I9Ri4QU9I4crTsHfSjQJKMGEuATyZJLgqzmW1uZNNFzpsGAZm9z2rCLWETZKBhgN12wZzxhn2uaz1No%2Fidx%2FtISJKynNyyYHdKsxkSi00o0p%2BiSxLMoeEI5mRYOyRySj%2FjkG7Z%2F7LBiL5wdrfZzxrtEq3vVzxbqcnbCm5jlPjRLmXpAfZQNLUuVY6GmvrheLVZTRlLnG7WEZAJBwgZLBDST%2FJe0D1yEhpJMaZtyu4GOL1AmchDmwgkZ%2Bi76iWs6Ou8XzaJb1YquJI4Kamf5fpRK78RBLx%2FUOs3GBQeWhu31Ivfqexad2IMEBPjvSkN4fahiyWRKyRgkRB94TuGt%2BhQjVSmfnGtnAysDFyERYR4jJn0gV8IjxF4K21UKSpPM59g3pAfVim5jVT0MXDItWihuosjvMb9tc75UsYA7plIGGOTgQBWvYsNZMbFyWHNjg5Yt7O7T2INCRFSkyDvd%2BpyX%2FIS0ScI%2BNX9mQcU47moXxgQpU10ViWwAhhcF8d4mn6a6sTPq9lcGSa%2BxGlBr1Mxb0hgUmSCSjN1Gas5sT5IV%2FNigBsPuaV%2BEHY33sUhxZTbcAW%2F48z6W%2FQOn1KvEigv%2Fu6NP3ucjoQotMNRRrEhuhTeoyPFfUhUNFdXjBj316d7wqklAxKC7zMokiNiwjtRbsZxr6E1CVF36PaWZ2YWm54J8agPWDlARu5aVxs3YdDiHLc5jjByHxsXv57JD1et5HYly%2FFMjwQrcWYN7Kh%2FXcdNIvaXSeO6AiD8a98wsaDW1AY6mAFT3bh2km6C12HqjfNmqXkDIvcOWHeKz6fhzk47Xp7VbRmxmd0X0HvWks3O3UhNW5T6%2B6GeJvYSVo2aRULp0ZE%2BCs9%2F%2F08shfGfYXdzRUN0gryDu0IDlEoiEncfY5SwyQSkKGRpu86RUy%2BBenrLvTmyVSZuifGuqbptf4oGMpxQwCZrsDUEN9ZfwXvnO7dDFRB4T5DSkLoHcw%3D%3D&Expires=1788190212)

### 1.3 보고 단위를 명시했다

새 버전은 계수를 쌍 단위로 계산한 뒤 개념별로 평균하고, 다시 42개 개념에 대해 평균한다고 설명한다. 또한 \(|\alpha|\)가 “쌍별 절대값의 평균”이라는 점도 명시했다.

이 수정은 이전 버전의 통계적 모호성을 상당히 줄인다. 2×2 정확도가 전체 2,480쌍을 합친 값이라는 설명도 독자가 macro와 micro 평균을 구분하는 데 도움이 된다.

다만 뒤에서 설명하듯, 이 정의만으로는 아직 계층적 의존성 문제가 완전히 해결되지는 않는다.

### 1.4 16.67% 기준선의 약점을 보완했다

새 버전은 단순히 “네 점수의 순서가 무작위라면 16.67%”라고 주장하지 않고, 실제 데이터에서 개념과 이미지-캡션 결합을 끊은 귀무 블록을 구성하여 0.67%라는 경험적 기준선을 보고한다.

이것은 중요한 개선이다. 실제 CLIP 점수는 독립적이고 교환 가능한 네 값이 아니기 때문에, 이론적 무작위 기준선만 제시하는 것보다 훨씬 타당하다.

다만 이 귀무 블록의 생성 방식은 더 자세히 설명되어야 한다.

- 다른 개념의 캡션을 어떤 비율로 결합했는가?
- 개념 난이도와 객체 빈도를 매칭했는가?
- 이미지와 캡션의 길이·템플릿·배경 분포를 통제했는가?
- 0.67%의 신뢰구간은 얼마인가?
- 귀무 블록 자체의 \(\alpha,\beta,\gamma\) 분포는 어떻게 되는가?

현재는 기준선의 존재는 설득력 있지만, 0.67%가 정말 “주효과 지배가 예측하는 귀무”인지까지는 충분히 검증되지 않았다.

### 1.5 개념 비의존 변환을 추가한 점이 가장 중요하다

이전 버전에서 가장 큰 약점은 개념별 프로브와 채점 규칙이 개념 정보를 이용할 가능성이었다. 새 버전은 이를 보완하기 위해 \(S_W=v^\top Wt\)를 사용하고, 개념 단위 GroupKFold로 채점되는 개념이 학습에 등장하지 않도록 했다.

이 실험은 논문의 설득력을 크게 높인다. 개념별 프로브가 특정 객체의 이름이나 데이터셋 편향을 외운 결과가 아니라, 새로운 개념에도 적용되는 공유된 채점 구조일 가능성을 보여주기 때문이다.

특히 다음 결과는 인상적이다.

- rank 2부터 개념 매크로 CI 하한이 16.67%를 넘음.
- rank 32에서 29.19%, 코사인의 7.2배.
- 무작위 변환은 0.44%.
- 라벨을 섞은 프로브 기반 변환은 0.69%.
- 파라미터 수가 훨씬 많은 완전 \(W\)는 23.91%로 rank 32보다 낮음.

이 결과는 “단순히 자유도가 많으면 좋아진다”는 설명보다 “적절한 저계수 구조가 중요하다”는 해석을 지지한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/83eeeca3-4d56-4fb7-9eaf-c45ec006e6c1/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE3ZGD6L3K&Signature=1j2QPBoxysCfu0bZxRiLaD8bTQg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFpDcHpJy1SxBA6ZwKtyz7OQpbE1LAtj4DRgn7%2BAe9gNAiEAyQo5hfPXk5ADCgHhILXAoWvfZevUzxM8LQQ%2FqkTwE5Mq%2FAQImP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDPXYdCmLDDYNxTu8iCrQBJRw0KBp8h0UC28OSXIZxLEIe3I9Ri4QU9I4crTsHfSjQJKMGEuATyZJLgqzmW1uZNNFzpsGAZm9z2rCLWETZKBhgN12wZzxhn2uaz1No%2Fidx%2FtISJKynNyyYHdKsxkSi00o0p%2BiSxLMoeEI5mRYOyRySj%2FjkG7Z%2F7LBiL5wdrfZzxrtEq3vVzxbqcnbCm5jlPjRLmXpAfZQNLUuVY6GmvrheLVZTRlLnG7WEZAJBwgZLBDST%2FJe0D1yEhpJMaZtyu4GOL1AmchDmwgkZ%2Bi76iWs6Ou8XzaJb1YquJI4Kamf5fpRK78RBLx%2FUOs3GBQeWhu31Ivfqexad2IMEBPjvSkN4fahiyWRKyRgkRB94TuGt%2BhQjVSmfnGtnAysDFyERYR4jJn0gV8IjxF4K21UKSpPM59g3pAfVim5jVT0MXDItWihuosjvMb9tc75UsYA7plIGGOTgQBWvYsNZMbFyWHNjg5Yt7O7T2INCRFSkyDvd%2BpyX%2FIS0ScI%2BNX9mQcU47moXxgQpU10ViWwAhhcF8d4mn6a6sTPq9lcGSa%2BxGlBr1Mxb0hgUmSCSjN1Gas5sT5IV%2FNigBsPuaV%2BEHY33sUhxZTbcAW%2F48z6W%2FQOn1KvEigv%2Fu6NP3ucjoQotMNRRrEhuhTeoyPFfUhUNFdXjBj316d7wqklAxKC7zMokiNiwjtRbsZxr6E1CVF36PaWZ2YWm54J8agPWDlARu5aVxs3YdDiHLc5jjByHxsXv57JD1et5HYly%2FFMjwQrcWYN7Kh%2FXcdNIvaXSeO6AiD8a98wsaDW1AY6mAFT3bh2km6C12HqjfNmqXkDIvcOWHeKz6fhzk47Xp7VbRmxmd0X0HvWks3O3UhNW5T6%2B6GeJvYSVo2aRULp0ZE%2BCs9%2F%2F08shfGfYXdzRUN0gryDu0IDlEoiEncfY5SwyQSkKGRpu86RUy%2BBenrLvTmyVSZuifGuqbptf4oGMpxQwCZrsDUEN9ZfwXvnO7dDFRB4T5DSkLoHcw%3D%3D&Expires=1788190212)

## 2. 여전히 남은 주요 문제

### 2.1 rank 변환의 학습 및 선택 절차가 핵심적으로 불명확하다

새 버전의 가장 중요한 결과가 rank-2부터 rank-32까지의 개념 비의존 bilinear 변환인데, 정작 \(W\)를 어떻게 학습했는지가 충분히 설명되지 않는다.

반드시 명시해야 할 내용은 다음과 같다.

- \(W\)의 학습 목적함수.
- rank 제약을 어떤 방식으로 적용했는가.
- \(W=UV^\top\) 형태라면 \(U,V\)의 차원.
- 최적화 알고리즘과 학습률.
- 학습 epoch와 early stopping 기준.
- temperature, bias, 정규화 사용 여부.
- 학습 개념과 평가 개념의 분할.
- rank 선택이 검증셋에서 이루어졌는지.
- rank 32가 사전에 정해진 값인지, 결과를 본 뒤 선택되었는지.
- 여러 rank를 시험한 데 대한 다중비교 또는 선택 편향 처리.

현재 본문은 “GroupKFold로 채점한다”고 말하지만, “채점”이 정확히 \(W\)의 학습까지 포함하는지, 아니면 고정된 \(W\)를 평가하는지 불분명하다. 이 부분은 논문의 핵심 결과를 재현할 수 있을 정도로 작성되어야 한다.

### 2.2 “개념 비의존”이라는 표현을 더 엄밀히 해야 한다

평가 개념이 학습에 등장하지 않는다고 해서 모든 형태의 개념 누수가 제거되는 것은 아니다. 42개 개념이 공유하는 다음 정보가 있을 수 있다.

- COCO의 객체 카테고리 체계.
- 유사한 객체 빈도.
- 템플릿과 문장 구조.
- 인페인팅 방식.
- 특정 개념군의 이미지 스타일.
- 이미지와 텍스트의 상태 라벨 구성.

따라서 “개념 비의존”보다는 **개념 수준으로 분리된 공유 bilinear 채점기** 또는 **held-out concept generalization**이라고 표현하는 편이 더 정확하다.

또한 새로운 개념의 평가에서 단순히 객체 단어가 바뀌는 것인지, 문장 템플릿과 이미지 분포까지 함께 바뀌는지 설명해야 한다.

### 2.3 rank-1과 rank-32 결과 사이의 해석이 아직 불안정하다

초록과 서론에서는 “같은 임베딩을 채점 규칙만 바꿔 읽으면 코사인의 7배”라고 강조하지만, 본문에서는 개념별 프로브의 결과가 26.13%, rank-1은 24.96% 또는 그에 가까운 값이며, 개념 비의존 변환의 최선은 rank 32에서 29.19%다.

따라서 독자는 다음을 혼동할 수 있다.

- 6배: 개념별 두 비트 채점기 기준.
- 7배: 개념 비의존 rank-32 \(W\) 기준.
- 24.96%: 개념별 프로브 외적 기준인지, 다른 rank-1 학습 기준인지.
- 22.11%: 학습된 rank-1 헤드의 성능.
- 23.91%: 완전 \(W\) 성능.

이 수치들을 하나의 표로 정리해야 한다.

| 채점기 | 개념 정보 사용 | 학습 구조 | 성능 |
|---|---:|---|---:|
| 코사인 | 없음 | 고정 \(W=I\) | 4.03% |
| 두 비트 \(f\) | 사용 | 프로브 기반 | 26.13% |
| 프로브 외적 rank-1 | 사용 | 고정 외적 | 24.96%로 보이는 결과 |
| 학습 rank-1 | 불명확 | bilinear | 22.11% |
| 개념 비의존 rank-32 | 없음 | GroupKFold | 29.19% |
| 완전 \(W\) | 없음 | full-rank | 23.91% |

실제 수치와 명칭을 정확히 정리하지 않으면, 논문의 메시지가 “rank-1이면 충분하다”인지 “rank-32가 가장 좋다”인지 불분명해진다.

### 2.4 rank-1에 대한 표현을 수정해야 한다

이전 버전의 “필요한 것은 rank 1”이라는 표현은 새 버전에서 완화되었지만, 여전히 결론 부분의 표현이 조심스럽지 않으면 과장으로 읽힐 수 있다.

현재 결과가 보여주는 것은 다음 중 하나일 수 있다.

1. rank-1만으로도 코사인보다 크게 개선된다.
2. rank-1 구조가 프로브의 극성 외적을 잘 표현한다.
3. 개념 비의존 일반화에서는 rank 2 이상이 더 안정적일 수 있다.
4. 최적 rank는 아직 판정되지 않았다.

실제로 본문은 rank 2부터 32까지 신뢰구간이 겹친다고 인정한다. 따라서 다음과 같이 쓰는 것이 정확하다.

> “저계수 bilinear 채점만으로도 고정 코사인보다 크게 개선되며, rank-1은 프로브 기반 극성 상호작용의 유효한 최소 모델이다. 다만 개념 비의존 조건에서 최적 rank는 아직 결정되지 않았다.”

즉, “필요한 것은 rank 1”보다 **“고정된 항등 채점보다 저계수 상호작용 채점이 중요하다”**가 현재 증거에 더 잘 맞는다.

### 2.5 이미지 인페인팅 아티팩트 문제가 여전히 가장 큰 경험적 약점이다

논문은 한계에서 자연 부재 이미지와 비교하지 않았다고 명시한다. 이는 정직한 서술이지만, 결론의 강도를 제한한다.

이미지 프로브가 존재 정보를 검출한다는 결과와 \(\beta\)가 큰 결과가 모두 인페인팅 흔적에 의해 설명될 수 있다. 예를 들어 모델이 “pizza가 사라졌다”를 표현한 것이 아니라 “이 장면은 인페인팅된 장면이다”를 감지했을 가능성이 있다.

최소한 다음 중 하나는 필요하다.

- 여러 인페인팅 방법 간 전이.
- 마스크 경계와 편집 흔적을 제거한 품질 통제.
- 자연적으로 객체가 없는 이미지의 보조 실험.
- 객체 영역이 아닌 배경 영역만 인페인팅한 대조군.
- 편집 아티팩트 분류기와 객체 부재 프로브의 성능 비교.

현재 버전에서 이 문제를 한계로 인정한 것은 좋지만, 핵심 결론인 “정보는 표현되어 있다”와 “교차항이 작다”가 모두 해당 데이터 생성 방식에 의존할 수 있다는 점은 여전히 심각하다.

### 2.6 프로브가 의미론적 정보를 검출한다는 주장은 여전히 제한적이다

어순 통제와 템플릿 전이는 개선되었지만, 텍스트 프로브가 실제로 “어느 객체가 부정되었는가”를 파악하는지에 대해서는 더 강한 검증이 필요하다.

예를 들어 모델은 다음을 이용할 수 있다.

- 문장 내 명사 위치.
- “features”와 “lacks”의 결합 패턴.
- 특정 객체가 특정 문장 위치에 나타나는 통계.
- 토큰의 위치와 구문적 주변 문맥.

“네 개 템플릿 계열을 가로지르는 전이가 95% 유지된다”는 수치는 긍정적이지만, 템플릿 계열의 구체적인 구성과 전이 방향이 제시되지 않아 해석하기 어렵다. 한 템플릿을 학습하고 완전히 다른 구문·어휘·어순에서 평가하는 교차 템플릿 표가 필요하다.

또한 이미지 프로브에는 어순에 대응하는 수준의 통제 실험이 없다. 이미지의 경우에도 객체 위치, 크기, 가림, 배경 구조를 통제해야 한다.

### 2.7 외부 과제의 예측 분석은 아직 탐색적이다

COCO 5,000장 갤러리에서 부호 있는 \(\beta\)가 부정 질의의 R@1 하락폭을 \(r=0.835\)로 예측한다는 결과는 흥미롭다. 반면 \(\gamma/|\beta|\)가 성적 자체를 예측하지 못한다는 결과도 논문의 주장을 정교하게 만든다.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/83eeeca3-4d56-4fb7-9eaf-c45ec006e6c1/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE3ZGD6L3K&Signature=1j2QPBoxysCfu0bZxRiLaD8bTQg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFpDcHpJy1SxBA6ZwKtyz7OQpbE1LAtj4DRgn7%2BAe9gNAiEAyQo5hfPXk5ADCgHhILXAoWvfZevUzxM8LQQ%2FqkTwE5Mq%2FAQImP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDPXYdCmLDDYNxTu8iCrQBJRw0KBp8h0UC28OSXIZxLEIe3I9Ri4QU9I4crTsHfSjQJKMGEuATyZJLgqzmW1uZNNFzpsGAZm9z2rCLWETZKBhgN12wZzxhn2uaz1No%2Fidx%2FtISJKynNyyYHdKsxkSi00o0p%2BiSxLMoeEI5mRYOyRySj%2FjkG7Z%2F7LBiL5wdrfZzxrtEq3vVzxbqcnbCm5jlPjRLmXpAfZQNLUuVY6GmvrheLVZTRlLnG7WEZAJBwgZLBDST%2FJe0D1yEhpJMaZtyu4GOL1AmchDmwgkZ%2Bi76iWs6Ou8XzaJb1YquJI4Kamf5fpRK78RBLx%2FUOs3GBQeWhu31Ivfqexad2IMEBPjvSkN4fahiyWRKyRgkRB94TuGt%2BhQjVSmfnGtnAysDFyERYR4jJn0gV8IjxF4K21UKSpPM59g3pAfVim5jVT0MXDItWihuosjvMb9tc75UsYA7plIGGOTgQBWvYsNZMbFyWHNjg5Yt7O7T2INCRFSkyDvd%2BpyX%2FIS0ScI%2BNX9mQcU47moXxgQpU10ViWwAhhcF8d4mn6a6sTPq9lcGSa%2BxGlBr1Mxb0hgUmSCSjN1Gas5sT5IV%2FNigBsPuaV%2BEHY33sUhxZTbcAW%2F48z6W%2FQOn1KvEigv%2Fu6NP3ucjoQotMNRRrEhuhTeoyPFfUhUNFdXjBj316d7wqklAxKC7zMokiNiwjtRbsZxr6E1CVF36PaWZ2YWm54J8agPWDlARu5aVxs3YdDiHLc5jjByHxsXv57JD1et5HYly%2FFMjwQrcWYN7Kh%2FXcdNIvaXSeO6AiD8a98wsaDW1AY6mAFT3bh2km6C12HqjfNmqXkDIvcOWHeKz6fhzk47Xp7VbRmxmd0X0HvWks3O3UhNW5T6%2B6GeJvYSVo2aRULp0ZE%2BCs9%2F%2F08shfGfYXdzRUN0gryDu0IDlEoiEncfY5SwyQSkKGRpu86RUy%2BBenrLvTmyVSZuifGuqbptf4oGMpxQwCZrsDUEN9ZfwXvnO7dDFRB4T5DSkLoHcw%3D%3D&Expires=1788190212)

그러나 이 분석에는 다음 문제가 있다.

- 상관분석의 독립 단위가 9개 모델인지, 모델-개념 조합인지 불명확하다.
- \(p=0.005\)가 어떤 표본 수와 검정 방식에 기반하는지 알 수 없다.
- “하락폭”은 모델 전체의 검색 특성이나 객체 빈도와 함께 움직일 수 있다.
- \(\beta\)와 R@1 하락폭이 동일한 데이터에서 계산되었다면 순환적 분석일 가능성이 있다.
- 사전에 예측을 정의했는지, 결과를 본 뒤 선택했는지 알 수 없다.

외부 과제는 현재로서는 인과적 검증보다는 **일관성 검증 또는 탐색적 예측 분석**으로 표현하는 것이 안전하다.

## 3. 수학적·개념적 점검

### 3.1 2×2 항등식은 타당하지만 부호 규약을 더 명확히 해야 한다

분해식과 성공 조건은 자연스럽지만, \(\gamma\)가 음수일 때의 처리와 “긍정/부정” 상태의 방향을 본문 앞부분에서 표로 보여주는 것이 좋다.

예를 들어 다음 표를 추가하면 된다.

| | 텍스트 \(b=+1\) | 텍스트 \(b=-1\) |
|---|---:|---:|
| 이미지 \(a=+1\) | \(S_{++}\) | \(S_{+-}\) |
| 이미지 \(a=-1\) | \(S_{-+}\) | \(S_{--}\) |

또한 \(\gamma\)가 양수이면 올바른 대각 결합이 선호된다는 것을 명시해야 한다. 현재는 설명을 읽어야만 부호의 의미를 확인할 수 있다.

### 3.2 “교차항이 양수”와 “교차항이 충분히 큼”을 분리해야 한다

새 버전은 이 점을 상당히 잘 설명하지만, 결론에서 다시 강조할 필요가 있다.

- \(\gamma>0\): 이미지와 텍스트 상태의 올바른 결합에 유리한 상호작용이 존재함.
- \(\gamma>\max(|\alpha|,|\beta|)\): 그 상호작용이 실제 2×2 검색 순위를 결정할 만큼 강함.

이 두 조건은 다르다. 논문의 가장 중요한 개념적 메시지가 바로 이 차이이므로, Figure 1 또는 Figure 2에서 두 조건을 시각적으로 대비하면 좋다.

### 3.3 \(\cos(d_I,d_T)\) 해석은 여전히 주의가 필요하다

두 모달리티가 공유 임베딩 공간에 있으므로 \(d_I,d_T\)의 코사인을 계산할 수 있다는 설명은 이전보다 좋아졌다. 그러나 다음 사항은 여전히 설명이 필요하다.

- \(d_I\)와 \(d_T\)를 어떤 샘플 평균으로 계산했는가.
- 상태 차이 벡터를 이미지·텍스트 각각 어떻게 정의했는가.
- 임베딩을 정규화한 뒤 차이를 계산했는가.
- \(\gamma\)와의 관계가 항등식인지 근사식인지.
- 평균 차이 벡터와 프로브 법선이 왜 이 데이터에서 일치하는가.
- 다른 데이터셋이나 모델에서도 이 관계가 유지되는가.

특히 \(\gamma \approx \frac14\|d_I\|\|d_T\|\cos(d_I,d_T)\)는 가법 모델 또는 평균화 가정 아래의 근사로 보이므로, 그 유도 또는 적용 조건을 각주나 부록에 제공해야 한다.

## 4. 수정 우선순위

### 최우선

1. **bilinear \(W\) 학습 절차를 완전히 공개할 것.**
2. rank별 결과, 신뢰구간, 파라미터 수, 학습·평가 분할을 표로 제시할 것.
3. 인페인팅 아티팩트에 대한 최소 하나의 통제 실험을 추가할 것.
4. 외부 과제에서 상관분석의 표본 단위와 검정 방법을 명시할 것.
5. “rank 1이 필요하다”는 표현을 “저계수 bilinear 구조가 충분히 효과적이다”로 완화할 것.

### 중요

6. 개념 단위 GroupKFold의 정확한 절차와 rank 선택 방식을 기술할 것.
7. 귀무 블록 0.67%의 생성 절차와 신뢰구간을 보고할 것.
8. 프로브 기반 결과와 개념 비의존 bilinear 결과를 하나의 비교표로 정리할 것.
9. 텍스트와 이미지 프로브의 데이터 누수 통제를 대칭적으로 제시할 것.
10. 모든 주요 수치에 macro/micro, 평균/중앙값, bootstrap 단위를 명시할 것.

### 있으면 좋은 수정

11. 자연 이미지 부재 데이터를 후속 실험이나 한계 확장으로 언급할 것.
12. \(\alpha,\beta,\gamma\)의 개념별 분포를 표준오차와 함께 시각화할 것.
13. 객체 빈도와 개념 난이도에 따른 결과 분해를 추가할 것.
14. \(\gamma\)를 직접 증가시키는 개입 실험을 포함할 것.
15. 코드, 데이터 분할, 학습 설정을 공개할 것.

## 5. 최종 평가

새 버전은 이전보다 확실히 강하다. 가장 큰 개선은 논문의 주장을 다음과 같이 정제한 점이다.

> CLIP에 부정 관련 정보가 전혀 없는 것이 아니라, 그 정보가 코사인 유사도의 교차항으로 충분히 반영되지 않는다.

이를 뒷받침하기 위해 논문은 이제 단순 프로브 결과뿐 아니라, 경험적 귀무 기준선, 개념 비의존 GroupKFold, rank별 bilinear 변환, 외부 검색 과제를 제시한다. 이 추가 결과 덕분에 “프로브가 잘 되지만 코사인이 못 한다”는 단순한 데이터셋 내부 관찰을 넘어, **표현과 크로스모달 채점 함수 사이의 구조적 불일치**라는 해석이 훨씬 설득력 있게 되었다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/83eeeca3-4d56-4fb7-9eaf-c45ec006e6c1/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE3ZGD6L3K&Signature=1j2QPBoxysCfu0bZxRiLaD8bTQg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFpDcHpJy1SxBA6ZwKtyz7OQpbE1LAtj4DRgn7%2BAe9gNAiEAyQo5hfPXk5ADCgHhILXAoWvfZevUzxM8LQQ%2FqkTwE5Mq%2FAQImP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDPXYdCmLDDYNxTu8iCrQBJRw0KBp8h0UC28OSXIZxLEIe3I9Ri4QU9I4crTsHfSjQJKMGEuATyZJLgqzmW1uZNNFzpsGAZm9z2rCLWETZKBhgN12wZzxhn2uaz1No%2Fidx%2FtISJKynNyyYHdKsxkSi00o0p%2BiSxLMoeEI5mRYOyRySj%2FjkG7Z%2F7LBiL5wdrfZzxrtEq3vVzxbqcnbCm5jlPjRLmXpAfZQNLUuVY6GmvrheLVZTRlLnG7WEZAJBwgZLBDST%2FJe0D1yEhpJMaZtyu4GOL1AmchDmwgkZ%2Bi76iWs6Ou8XzaJb1YquJI4Kamf5fpRK78RBLx%2FUOs3GBQeWhu31Ivfqexad2IMEBPjvSkN4fahiyWRKyRgkRB94TuGt%2BhQjVSmfnGtnAysDFyERYR4jJn0gV8IjxF4K21UKSpPM59g3pAfVim5jVT0MXDItWihuosjvMb9tc75UsYA7plIGGOTgQBWvYsNZMbFyWHNjg5Yt7O7T2INCRFSkyDvd%2BpyX%2FIS0ScI%2BNX9mQcU47moXxgQpU10ViWwAhhcF8d4mn6a6sTPq9lcGSa%2BxGlBr1Mxb0hgUmSCSjN1Gas5sT5IV%2FNigBsPuaV%2BEHY33sUhxZTbcAW%2F48z6W%2FQOn1KvEigv%2Fu6NP3ucjoQotMNRRrEhuhTeoyPFfUhUNFdXjBj316d7wqklAxKC7zMokiNiwjtRbsZxr6E1CVF36PaWZ2YWm54J8agPWDlARu5aVxs3YdDiHLc5jjByHxsXv57JD1et5HYly%2FFMjwQrcWYN7Kh%2FXcdNIvaXSeO6AiD8a98wsaDW1AY6mAFT3bh2km6C12HqjfNmqXkDIvcOWHeKz6fhzk47Xp7VbRmxmd0X0HvWks3O3UhNW5T6%2B6GeJvYSVo2aRULp0ZE%2BCs9%2F%2F08shfGfYXdzRUN0gryDu0IDlEoiEncfY5SwyQSkKGRpu86RUy%2BBenrLvTmyVSZuifGuqbptf4oGMpxQwCZrsDUEN9ZfwXvnO7dDFRB4T5DSkLoHcw%3D%3D&Expires=1788190212)

하지만 현 단계에서 가장 조심해야 할 것은 **결과의 해석 강도**다. 논문은 “교차항이 존재하지만 작다”는 점은 잘 보여주지만, 그것이 보편적인 CLIP 메커니즘인지, 인페인팅 데이터의 특수성인지, 또는 특정 템플릿과 객체 조합의 결과인지는 아직 완전히 분리하지 못했다. 또한 rank-32 결과가 실제로 rank-1보다 우월한지, bilinear 변환의 일반화가 어떤 조건에서 성립하는지도 더 명확히 해야 한다.

따라서 현재 판정은 다음과 같다.

**판정: Weak Accept에 가까워진 Major Revision**

- 아이디어: 강함.
- 문제의식: 명확함.
- 수학적 분석: 정확하지만 자체로는 표준적임.
- 실증 결과: 흥미롭고 상당히 개선됨.
- 실험 설계: 이전보다 강해졌으나 인페인팅·프로브·bilinear 학습 절차가 아직 핵심 약점.
- 주장 범위: 대체로 적절하나 rank-1 및 일반화 표현은 완화 필요.
- 재현성: 본문 정보만으로는 부족함.

국내 학술대회 2페이지 논문으로는 충분히 경쟁력이 있으며, 상세 부록이나 완결 논문에 실험 설정과 통계 검증을 보강한다면 **명확한 Accept 수준**까지 올라갈 수 있다.

**비판적 리뷰: “다객체 장면의 객체 존재 부정: 표현된 정보와 유사도 상호작용의 괴리”**

### 요약
이 논문은 CLIP 계열 이중 인코더가 다객체 장면에서 객체 부재를 요구하는 질의에서 반복적으로 실패하는 현상을 다룬다. 저자들은 2×2 매칭 점수를 상수 + 이미지 주효과(β) + 텍스트 주효과(α) + 교차항(γ)으로 분해하고, 부정 검색 성공이 정확히 \(\gamma > \max(|\alpha|,|\beta|)\)와 동치임을 보인다. BEAF 반사실 이미지 쌍과 AB-swap 텍스트 쌍(42개 개념, 2,480쌍)에서 측정한 결과, 선형 프로브로는 정보가 검출되지만(존재 탐지 macro AUC 0.759, 어순 통제 후 텍스트 프로브 64.69%), γ는 지배 주효과의 약 1/5.2에 불과해 2×2 정답률이 4.03%에 머문다. 같은 임베딩을 채점 규칙만 바꿔 읽으면(개념 비의존 rank-32 변환 포함) 코사인의 7배 수준까지 회복되며, 이 패턴은 아홉 모델 전반에서 일관된다. 외부 T2I 검색에서도 β가 부정 질의로 인한 R@1 하락폭을 잘 예측한다.

### 강점
1. **형식화의 명확성**  
   분해 자체가 표준 요인 설계 도구임을 명시하고, 기여를 “실측”에 둔 점이 이전 버전보다 정직하고 좋다. 성공 조건이 항등식으로 성립한다는 점은 진단의 힘을 크게 높인다.

2. **실험 설계의 통제**  
   BEAF 인페인팅 쌍 + AB-swap 텍스트는 토큰 다중집합을 거의 동일하게 유지하면서 “어느 객체가 부정되었는가”를 강제한다. 어순 통제 실험과 템플릿 전이 실험으로 표면 단서의 기여를 정량화한 점도 성실하다.

3. **귀무 가설과 회복 실험의 개선**  
   개념을 섞어 만든 블록에서 0.67%라는 현실적 귀무를 제시한 것은 좋다. 개념 비의존 GroupKFold 설정에서 rank-32가 29.19%(코사인의 7.2배)에 도달하고, 무작위/라벨 섞은 통제가 오히려 코사인보다 낮은 결과를 보인 점은 “필요한 것은 올바른 저계수 구조”라는 주장을 뒷받침한다.

4. **다모델 일반화와 외부 검증**  
   아키텍처·데이터·목적함수·부정 특화 미세조정을 아우르는 9개 모델에서 378개 조합 중 성공 조건 충족이 0개라는 결과는 강력하다. COCO 갤러리에서 β가 R@1 하락폭을 \(r=+0.835\)로 예측한다는 외부 검증도 설득력 있다.

5. **선행 연구와의 정리**  
   [3]과 [4]의 주장을 같은 분해의 서로 다른 항으로 해석하는 부분은 깔끔하다.

### 주요 약점 및 우려

**1. 범위의 한계가 여전히 크다**  
제목과 서론에서 “객체 존재 부정”으로 범위를 명확히 한 것은 개선되었으나, 관계·행위 부정, 단일 객체 “no-X” 질의, 자연적으로 객체가 없는 이미지와의 비교는 여전히 프레임 밖이다. 인페인팅 아티팩트가 존재/부재 신호에 영향을 줄 가능성을 완전히 배제하기 어렵다. 한계 섹션에서 이를 인정한 점은 좋지만, 주장의 일반화 범위는 여전히 좁다.

**2. AB-swap의 잔여 단서**  
어순을 통제하면 73.80% → 64.69%로 떨어진다. 저자들은 “대부분은 표면 어순이 아니다”라고 주장하지만, 12% 상대 기여는 무시하기 어렵다. 토큰 다중집합과 위치까지 완전히 맞춘 더 엄격한 통제, 또는 다양한 템플릿에서의 일관성을 더 자세히 보고할 필요가 있다.

**3. 회복 실험의 해석**  
개념 비의존 rank-k 실험은 이전 버전의 가장 큰 약점을 상당 부분 해결했다. 그러나 rank-2부터 이미 우연을 넘고 rank-32가 최선이라는 결과는, “rank-1이면 충분하다”는 초기 직관과는 다소 어긋난다. 최적 rank가 결정되지 않았고 신뢰구간이 겹친다는 한계 인정이 있긴 하나, “필요한 구조의 복잡도”에 대한 논의가 더 깊어질 수 있다. 또한 이 변환이 실제 대규모 검색에서 어떻게 작동할지, 개념 비의존적으로 어떻게 학습·적용할지는 미지수로 남는다.

**4. 수치 보고와 통계적 엄밀성**  
- 계수 평균 방식(|α|는 쌍별 절대값의 평균 등)을 명시한 것은 개선이다.  
- γ의 부트스트랩 CI와 Wilcoxon 결과는 설득력 있으나, 다중 비교 보정이나 개념 간 분산 구조에 대한 더 자세한 보고가 있으면 좋다.  
- “비트 단위 100% 일치”는 구현 검증으로는 충분하나, 본질적 기여는 아니다.

**5. 기계론적 설명의 부족**  
\(\cos(d_I, d_T) \approx 0.167\)이고 이것이 γ와 상관(\(\rho=+0.742\))된다는 기하학적 관찰은 흥미롭다. 그러나 *왜* 두 모달리티의 극성 방향이 이렇게 정렬되지 않는지에 대한 설명(대조 학습의 최적화 특성, 학습 데이터 분포, bag-of-words 경향 등)은 여전히 얕다. 진단은 잘 되었으나 원인 규명은 다음 단계로 남겨져 있다.

**6. 실용적 함의**  
병목이 “표현이 아니라 고정된 항등 행렬 메트릭”임을 보인 것은 가치 있으나, 실제 시스템에 적용 가능한 해결책(예: 학습 가능한 저계수 헤드를 개념 라벨 없이 어떻게 얻을 것인가)에 대한 논의가 부족하다.

### 전체 평가
이전 버전에 비해 제목·범위 명시, 기여 위치 정리, 귀무 가설, 개념 비의존 회복 실험, 한계 서술이 모두 개선되었다. 핵심 통찰—“정보는 선형적으로 검출되지만 교차항이 주효과에 밀려 순위를 뒤집지 못한다”—은 명확하고 재사용 가능한 진단 도구를 제공한다.

다만 범위가 다객체 장면의 객체 존재 부정에 국한되고, 잔여 표면 단서와 기계론적 설명, 실제 적용 경로에 대한 논의가 아쉽다. 짧은 논문/워크숍 논문으로는 충분히 강하며, 탑티어 학회에서는 범위 확장이나 더 엄격한 통제, 원인에 대한 추가 분석이 있으면 설득력이 더 커질 것이다.

**권고**: Minor revision 후 수용 가능. 핵심 형식화와 실측은 공유할 가치가 충분하다.

제공해주신 "PAPER_2PAGE.pdf" 논문에 대한 리뷰입니다. 노트북에 포함된 다양한 최신 VLM(Vision-Language Model)의 부정(Negation) 이해 관련 문헌들을 함께 참고하여 작성했습니다.

---

### **[논문 요약]**

이 논문("PAPER_2PAGE.pdf")은 CLIP 계열 이중 인코더 모델이 다객체 장면에서 객체의 부재(negation)를 요구하는 질의에 실패하는 원인을 수학적, 실험적으로 분석한 연구입니다.

* **방법론:** 저자들은 BEAF 기반의 반사실적(counterfactual) 이미지 쌍과, 단어 집합은 동일하되 결합만 뒤바꾼 'AB-swap' 텍스트 쌍을 활용하여 2×2 최소쌍을 구축했습니다. 크로스모달 매칭 점수($S_{ab}$)를 이미지 주효과($\beta$), 텍스트 주효과($\alpha$), 그리고 교차항($\gamma$)으로 요인 분해(Factorization)하여, 부정 검색의 성공 조건이 $\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$라는 대수적 부등식과 동치임을 도출해 냈습니다.


* **실험 결과:** 선형 프로브를 통해 텍스트 내 부정 정보 자체는 검출(어순 통제 후 64.69%, 존재 탐지 AUC 0.759)되지만, 교차항($\gamma$)의 크기가 지배 주효과의 1/5.2에 불과해 기존 코사인 매칭에서는 정답률이 4.03%에 그친다는 점을 밝혔습니다. 반면, 임베딩을 유지한 채 채점 규칙만 랭크 32(rank-32) 형태의 변환으로 바꾸면 정답률이 29.19%로 코사인의 약 7.2배 상승함을 보여주었습니다.


* **결론:** VLM이 부정에 실패하는 이유는 정보가 없어서가 아니라, 코사인 유사도라는 채점 규칙이 필요한 저계수 구조(low-rank structure)를 항등 행렬로 고정해버리기 때문임을 규명했습니다.



### **[강점 (Strengths)]**

* **정교하게 통제된 최소쌍 실험 설계:** 단순한 부정 표지("no", "without")의 유무로 데이터를 나눈 것이 아니라, 어휘가 완전히 동일한 AB-swap 쌍과 특정 객체 하나만 인페인팅으로 제거한 배경 공유 이미지 쌍을 사용하여 모델이 '표면적 단어의 존재'만으로 정답을 유추할 수 없도록 완벽히 통제했습니다.


* **실패 원인의 명확한 대수적 규명:** VLM의 부정 처리 실패라는 추상적 문제를 '교차항 크기 부족'이라는 측정 가능한 부등식 조건($\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$)으로 정량화하여 원인을 수학적으로 명확히 입증했습니다.


* **표현과 채점 규칙의 분리 및 학술적 통합:** 기존 연구들이 제기한 "정보가 없다"는 주장과 "결합 정보는 있다"는 모순된 주장들을 요인 분해의 서로 다른 항($\alpha, \beta$ 지배 현상 vs $\gamma \neq 0$)으로 묶어내어 통합적 시각을 제공했습니다. 실패의 병목이 임베딩 표현(representation) 자체에 있는 것이 아니라 두 모달리티의 극성 정렬 문제와 코사인 채점 규칙에 있음을 증명한 점은 학술적 가치가 매우 높습니다.



### **[약점 및 개선 제안 (Weaknesses & Suggestions)]**

* **적용 범위의 한계 (Scope Limitations):** 저자들도 언급했듯, 이 연구는 '다객체 장면의 객체 존재 부정'에만 초점을 맞추고 있습니다. 행위 부정, 관계 부정, 혹은 "a photo of no cat"과 같이 결합할 상대가 없는 단일 객체 부정에 대해서는 텍스트 상태 $b$가 정의되지 않아 분석 프레임이 적용되지 않는다는 한계가 있습니다. 향후 다른 유형의 부정으로 확장할 논리적 기반을 부록에라도 제안하면 좋을 것입니다.


* **자연 이미지에 대한 검증 부재:** 본 연구의 이미지는 모두 인페인팅을 통해 인위적으로 만들어진 반사실적 이미지입니다. 원래부터 특정 객체가 존재하지 않는 자연 이미지(Natural images) 환경에서도 이 요인 분해 현상과 rank-32 채점 함수의 우위가 동일하게 유지되는지 추가 검증이 필요합니다.


* **최신 추론 타임(Test-time) 방법론들과의 비교:** 논문에서는 9종의 모델(부정 특화 미세조정 4종 포함)이 모두 성공 조건을 만족하지 못했다고 언급했습니다. 그러나 최근 학계에서는 NegBench, ConCLIP 등의 미세조정 방식의 한계를 극복하기 위해, Test-time adaptation을 수행하는 NEAT나 부분 공간(Sub-space)을 기하학적으로 모델링하는 SpaceVLM과 같이 모델 파라미터 업데이트 없이 부정을 해결하려는 시도들이 활발히 등장하고 있습니다. 본 논문에서 제안한 rank-32 변환 채점기가 이러한 최신 Inference-time 보정 기법들과 비교하여 실제 검색/분류 파이프라인에서 어떤 비교 우위나 차이점이 있는지 논의(Discussion) 섹션에 추가한다면 연구의 실용적 임팩트가 훨씬 커질 것입니다.



### **[총평 (Overall Evaluation)]**

본 논문은 최근 VLM 연구의 주요 난제인 '긍정 편향(Affirmative bias)' 및 부정 검색 실패의 근본 원인을 데이터의 부재가 아닌 임베딩 공간의 구조적 결함(채점 함수)에서 찾아낸 매우 통찰력 있는 연구입니다. 기존의 많은 데이터 중심(Data-centric) 접근법들이 맹목적으로 합성 데이터를 증강하여 파인튜닝을 시도해 온 것과 달리, 현상 이면의 수학적 원리를 깊이 파고들어 대안을 증명해냈다는 점에서 학계에 큰 기여를 합니다. 위에서 언급한 한계점들을 논의 및 향후 과제로 일부 보완한다면, 의심의 여지 없이 훌륭한 논문이 될 것입니다. **강력히 게재를 추천(Strong Accept)합니다.**

학회의 평판을 최우선으로 고려하는 엄격한 리뷰어의 관점에서, 제출된 "PAPER_2PAGE.pdf" 논문과 노트북에 제공된 참고 문헌들을 종합적으로 검토한 리뷰 결과를 전달해 드립니다.

본 논문은 Vision-Language Models (VLM)이 부정(Negation)을 이해하지 못하는 원인을 깊이 있게 분석한 흥미로운 연구입니다. 그러나 최고 수준의 학회에 억셉트되기 위해서는 선행 연구와의 연결성 및 방법론적 확장성 측면에서 보완해야 할 몇 가지 중요한 약점들이 존재합니다.

---

### 📝 논문 요약 및 총평

"PAPER_2PAGE.pdf" 논문은 CLIP 계열 모델이 "개가 없는 사진"과 같이 객체의 부재를 묘사하는 질의에서 실패하는 원인을 2x2 요인 설계 분해(Factorial decomposition)를 통해 분석했습니다. 저자들은 모델이 부정에 대한 정보를 전혀 학습하지 못한 것이 아니라(텍스트 프로브를 통해 정보 검출 확인), 최종 코사인 유사도 기반의 크로스모달 인터페이스에서 상호작용 항(교차항)이 텍스트 및 이미지의 주효과를 압도하지 못해 순위를 뒤집지 못한다는 점을 수학적 및 경험적으로 증명했습니다.

이 논문은 현상의 표면적 실패를 넘어 '정보의 부재'와 '유사도 연산 구조의 한계'를 구분했다는 점에서 학술적 가치가 높습니다. 하지만 최신 연구 동향을 고려할 때 아래와 같은 비판적 개선점이 요구됩니다.

### 👍 강점 (Strengths)

* **정량적 원인 규명의 독창성:** 기존 연구들이 단순히 모델이 부정을 이해하지 못한다고 평가한 것에 반해, 이 논문은 매칭 점수를 주효과와 교차항으로 분해하여 교차항의 크기가 지배 주효과의 1/5.2에 불과함을 측정 가능한 수치로 증명해 냈습니다.


* **구조적 병목 현상 증명:** 코사인 유사도 규칙 하에서는 2x2 정답률이 4.03%에 불과하지만, 동일한 임베딩을 기반으로 2비트 합성이나 rank 32 변환 등의 다른 채점 규칙을 적용했을 때 정답률이 코사인의 약 7.2배인 29.19%까지 도달함을 보여주었습니다. 이는 문제의 병목이 임베딩 표현 자체가 아니라, 유사도를 계산하는 구조(채점 규칙)에 있음을 명확히 짚어낸 훌륭한 기여입니다.


* **철저한 변인 통제:** AB-swap 쌍을 사용하여 토큰 다중집합을 통제하고, 어순 통제 후에도 64.69%의 텍스트 프로브 정확도를 확보하여 모델 내부에 실제 정보가 존재함을 입증한 실험 설계가 탄탄합니다.



### 👎 약점 및 개선 요구사항 (Weaknesses & Revisions)

수준 높은 논문이 되기 위해 다음 사항들을 수정 및 보완해야 합니다.

* **기하학적 한계 증명(Geometric Impossibility)과의 연결 부족:**
* 본 논문은 코사인 매칭을 항등 행렬로 고정해 둔 것이 문제라고 지적합니다.


* 그러나 최근 연구에서는 단위원 벡터 임베딩에 대한 단순 코사인 유사도가 속성 결합, 공간 관계 및 부정을 정확하게 표현하는 데 근본적인 기하학적 제한이 있다는 수학적 증명을 제시한 바 있습니다.


* 논문의 요인 분해 주장을 강화하기 위해, 이러한 기하학적 불가능성 정리(Geometric impossibility theorem)를 인용하고 코사인 기반 유사도의 근본적 한계를 보다 수학적인 맥락에서 논의해야 합니다.




* **임베딩 수정 및 조향(Steering) 방식과의 비교 부재:**
* 저자들은 고정된 rank-32 변환이나 외부 채점기를 도입하여 문제를 해결할 수 있다고 주장합니다. 이는 CLIP이 교차 모달리티에서는 bag-of-words처럼 행동하지만 단일 모달리티에서는 결합 정보를 가지고 있으며, 텍스트 임베딩에 대한 단순 선형 변환으로 이를 개선할 수 있다는 기존 발견과 일맥상통합니다.


* 반면, 최근 연구들은 새로운 채점 규칙을 학습하거나 도입하지 않고도, 텍스트 잠재 공간(latent space)에 이미 존재하는 부정 방향(negation direction)을 추출하여 테스트 타임에 조향(test-time steering)하는 방법으로 부정을 성공적으로 처리할 수 있음을 증명했습니다.


* 또한 방향성 오프셋(directional offset)을 사용해 구성적 산술 연산으로 의미 정보를 직접 제거하는 Zero-shot 임베딩 교정 방식도 제안되었습니다.


* 리뷰어로서 질문합니다: 본 논문이 제안하는 '채점 규칙의 변경'이 이러한 테스트 타임 임베딩 개입(Intervention) 방식보다 실용적이거나 이론적으로 더 우월한 이유는 무엇입니까? 이에 대한 논의가 추가되어야 합니다.


* **평가 환경의 제한성과 데이터 중심 접근법에 대한 논의 누락:**
* 본 논문의 한계점에서도 언급되었듯, 이 연구는 다객체 장면에서의 객체 존재 부정만을 다루고 있으며 단일 객체 부정이나 관계 부정은 배제되었습니다.


* VLM 모델들은 언어적 사전 지식(linguistic priors)에 과도하게 의존하는 경향이 있으므로, 편향을 배제하고 다양한 템플릿과 상황에 대한 평가가 필수적입니다.


* 더불어 논문은 표현 구조의 한계를 지적하지만, 최근 일부 연구는 다양한 언어적 스타일을 포함한 명시적 부정 예제로 구성된 데이터셋을 파인튜닝(데이터 중심 접근)함으로써 모델의 부정 검색 능력과 객관식(MCQ) 능력을 의미 있게 향상시켰음을 보여줍니다. 코사인 유사도의 한계에도 불구하고 데이터 확장이 가져오는 실질적 성능 향상에 대해서도 고찰할 필요가 있습니다.





### ⚖️ 최종 판정 (Recommendation)

**Weak Accept (수정 후 승인)**

"PAPER_2PAGE.pdf"는 현상의 근본 원인을 '구조적 제약(코사인 유사도의 한계)'으로 수치화하여 밝혀냈다는 점에서 양질의 통찰을 제공합니다. 다만, 학회의 높은 기준을 충족하기 위해서는 논문의 후반부 Discussion 섹션에서 상기 지적된 최신 연구들(기하학적 제약 증명 및 테스트 타임 개입 기법)과의 관계를 명확히 하고, 제안한 진단 도구가 갖는 차별성을 더 강하게 어필하는 방향으로의 **Minor Revision**이 반드시 선행되어야 합니다.

리뷰어 관점에서 상당히 엄격하게 보겠습니다. 결론부터 말하면, 현재 버전은 **흥미로운 진단 논문이지만, 주요 학회에서 accept를 주기에는 아직 증거가 부족합니다.** 제 판단은 **Reject에 가깝고, 약간의 수정으로 해결되는 수준을 넘어 핵심 주장에 대한 검증이 더 필요합니다.**

다만 아이디어 자체가 약하다는 뜻은 아닙니다. 오히려 현재 프로젝트에서 가장 발전 가능성이 있는 부분은 분명히 있습니다. 문제는 논문의 가장 강한 문장이 현재 실험으로는 그만큼 강하게 입증되지 않는다는 것입니다.

## 1. 논문이 실제로 주장하는 것

논문의 핵심 논리는 상당히 명확합니다.

CLIP의 2×2 score matrix를

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

로 분해하고,

* \(\alpha\): text main effect
* \(\beta\): image main effect
* \(\gamma\): image–text interaction

으로 해석합니다.

그리고

$$
\Delta
=
\min(S_{++},S_{--})
-
\max(S_{+-},S_{-+})
$$

에 대해

$$
\Delta
=
2\gamma-2\max(|\alpha|,|\beta|)
$$

라는 항등식을 이용합니다.

따라서 2×2 matching이 성공하려면

$$
\gamma>\max(|\alpha|,|\beta|)
$$

여야 한다는 것입니다. 이 수학 자체는 타당합니다. 실제로 Winoground의 group score가 요구하는 것도 네 개의 image-caption pair 중 올바른 두 대각 성분이 잘못된 두 비대각 성분보다 모두 높게 나오는 것이므로, 이 구조와 직접 연결됩니다. Winoground 역시 동일한 단어 집합을 공유하는 twin captions를 사용하여 단순한 bag-of-words 전략을 차단하고 compositionality를 평가했습니다.  

논문은 여기서 중요한 empirical observation을 제시합니다.

$$
|\alpha|=0.00380,\quad
|\beta|=0.00628,\quad
\gamma=0.00121
$$

이고, 결과적으로 cosine 기반 2×2 matching accuracy가 **4.03%**에 불과하다고 보고합니다. 반면 negation-related information 자체는 linear probe에서 검출되며, image AUC 0.759, text pair accuracy 64.69%까지 나온다고 주장합니다. 

즉 논문의 핵심 메시지는

> **"negation information이 representation에 없는 것이 아니라, 그것이 cross-modal similarity를 통해 충분한 interaction으로 나타나지 않는다."**

입니다.

이 framing은 상당히 좋습니다.

---

# 2. 가장 좋은 점: 기존 연구와 구별되는 질문을 만들었다

이 부분은 인정해야 합니다.

기존 연구는 대략 세 방향으로 흘렀습니다.

첫째, NegBench 계열은 CLIP이 negation을 잘 못한다는 것을 대규모 benchmark로 보였습니다. CVPR 2025의 NegBench는 retrieval과 MCQ를 통해 negation failure를 평가하고, synthetic negation data로 fine-tuning했을 때 성능이 향상될 수 있음을 보였습니다. 

둘째, representation을 수정하는 연구가 있습니다. 예를 들어 CVPR 2026의 연구는 CLIP embedding space에 negation direction이 존재한다고 보고 representation engineering으로 이를 조작합니다. 

셋째, similarity 자체를 변경하거나 별도의 scoring mechanism을 사용하는 연구가 있습니다. `Is CLIP ideal?`은 더 근본적으로 CLIP의 joint embedding geometry가 attribute binding, spatial relation, negation 등을 동시에 표현하는 데 구조적 제약이 있다고 주장합니다. 

그리고 최근에는 intermediate representation에서 negation signal을 회수하려는 PeakPatch도 이미 등장했습니다. 이 논문은 middle layer에서 compositional signal이 존재하지만 final representation에서 collapse한다고 주장합니다. 

그렇다면 이 논문의 질문은 조금 다릅니다.

**"정보가 representation에 존재하는가?"**

가 아니라

**"representation에 존재하는 정보가 실제 CLIP의 cross-modal scoring decision을 지배할 정도로 강한가?"**

입니다.

이 distinction은 좋습니다.

특히 논문이 스스로 "분해 자체는 contribution이 아니다"라고 선을 긋는 것도 적절합니다. 

---

# 3. 하지만 가장 큰 문제: "information exists" → "similarity bottleneck"은 아직 성립하지 않는다

제가 리뷰어라면 가장 먼저 이 부분을 공격합니다.

논문은

> linear probe가 negation을 검출한다
> +
> cosine은 negation matching을 못한다

를 보여준 뒤,

> 따라서 representation에는 정보가 있지만 cosine interface가 그것을 제대로 사용하지 못한다

고 해석합니다.

하지만 이 추론에는 중요한 논리적 간극이 있습니다.

### Linear decodability와 cross-modal usability는 다른 문제입니다.

어떤 representation \(z\)에 정보 \(q\)가 선형적으로 존재한다고 합시다.

$$
q \approx w^Tz
$$

이것은 단지

> 어떤 별도의 decoder \(w\)를 학습하면 q를 복원할 수 있다

는 뜻입니다.

그러나 CLIP은

$$
S(I,T)=v_I^Tv_T
$$

라는 **고정된 decoder**를 사용합니다.

따라서

$$
\exists w:\quad w^Tz \text{ predicts negation}
$$

으로부터

$$
v_I^Tv_T \text{ can exploit negation}
$$

을 도출할 수 없습니다.

오히려 이 논문의 발견 자체가 정확히 이 차이를 보여줍니다.

즉 이 논문이 진짜로 증명하는 것은

> **"negation is linearly decodable but poorly exploited by the vanilla cross-modal scoring function."**

정도입니다.

이것은 충분히 흥미롭지만,

> **"the bottleneck is the similarity interface"**

라는 더 강한 causal claim은 아직 아닙니다.

---

# 4. 특히 3.3의 "7배 개선"은 생각보다 강한 증거가 아니다

논문에서 가장 눈에 띄는 결과가 이것입니다.

같은 embedding을 그대로 두고

$$
S_W=v^TWt
$$

라는 learned bilinear scoring으로 바꾸자 rank-32에서 29.19%가 나오고, cosine 4.03%의 약 7.2배가 된다는 것입니다. 또한 random rank-1은 0.44%, full-rank는 23.91%라고 보고합니다. 

이 결과는 분명 흥미롭습니다.

하지만 리뷰어라면 바로 묻습니다.

**"왜 이것이 negation-specific evidence인가?"**

\(W\)는 사실상 새로운 multimodal matching model입니다.

즉,

$$
v^Tv
$$

라는 identity matrix를

$$
v^TWt
$$

라는 학습 가능한 cross-modal transformation으로 교체한 것입니다.

그 결과가 좋아졌다는 것은

> **CLIP의 identity metric이 최적이 아니다**

라는 것을 보여줍니다.

하지만 이것이

> **negation이 특별히 similarity metric 때문에 실패한다**

는 것을 보여주지는 않습니다.

왜냐하면 \(W\)가 다음과 같은 것도 개선할 수 있기 때문입니다.

* object binding
* color binding
* spatial relation
* attribute binding
* compositionality
* modality gap
* 일반적인 image-text alignment

즉 현재 실험에서는 **W가 negation을 고친 것인지, compositional matching 전반을 고친 것인지 분리되지 않습니다.**

이건 상당히 중요한 문제입니다.

---

# 5. Winoground를 참고하면 이 약점이 더 명확해진다

Winoground의 중요한 철학은 단순히 "성능이 낮다"가 아닙니다.

**word content 자체를 통제해서 lexical shortcut을 제거하는 것**입니다.

실제로 Winoground는 두 caption이 동일한 words/morphemes를 가지고 순서만 달라지도록 구성했고, 그래서 bag-of-words 모델이 원리적으로 제대로 풀 수 없게 했습니다.  

그런데 현재 논문의 AB-swap은 이 철학을 부분적으로 가져왔지만, 완전히 동일한 수준의 통제는 아닙니다.

논문의 예는

> "The scene features a pizza, but lacks a cup."

와

> "The scene features a cup, but lacks a pizza."

입니다. 

여기서는 단어 집합은 거의 동일하지만 **syntax와 object-negation assignment가 동시에 바뀝니다.**

이것 자체는 좋은 설계입니다.

문제는 linear probe가 이것을 잘 푸는 것이

> "negation semantics가 embedding에 존재한다"

는 것과 정확히 같은 것은 아니라는 점입니다.

왜냐하면 모델이 학습할 수 있는 것은

* `pizza`가 `features`와 결합
* `cup`이 `lacks`와 결합

이라는 **lexico-syntactic association**일 수도 있기 때문입니다.

실제로 논문도 순서 control을 하면 text probe accuracy가 73.80%에서 64.69%로 떨어진다고 보고합니다. 즉 상당한 signal이 position에 의해 설명됩니다. 

저라면 이 결과를 오히려 더 강하게 파고들겠습니다.

---

# 6. 가장 심각한 실험적 문제: BEAF/inpainting artifact

이것은 제가 리뷰에서 **major concern**으로 적을 부분입니다.

이미지는 BEAF의 before/after image이고, 특정 object를 inpainting으로 제거합니다. 논문 스스로도 자연적으로 객체가 없는 이미지와 비교하지 않았음을 인정합니다. 

그러면 image embedding에서 linear probe가

$$
\text{present} \leftrightarrow \text{absent}
$$

를 구분하는 이유가 반드시

> "CLIP이 객체의 존재/부재를 semantic하게 표현하기 때문"

이라고 할 수 없습니다.

다음도 가능합니다.

$$
\text{original image}
\rightarrow
\text{inpainting artifact}
\rightarrow
\text{embedding difference}
$$

즉 probe가 실제로 학습하는 것은 object absence가 아니라 **inpainting signature**일 수 있습니다.

특히 같은 장면에서 object 하나만 제거했기 때문에 두 이미지가 매우 강하게 paired되어 있습니다.

따라서 image-side AUC 0.759는 생각보다 해석하기 어렵습니다.

논문의 기존 markdown에서도 실제로 이 문제가 꽤 심각하게 관찰되어 있습니다. 동일 장면 최소쌍에서 개념 혼합 시 AUC가 크게 감소하는 결과가 있었고, 특정 객체 제거가 similarity를 감소시키지 않는 사례도 상당수 있었습니다. 

이 부분을 해결하지 않고 "representation contains absence information"이라고 말하면 리뷰어에게 공격받을 가능성이 높습니다.

---

# 7. 더 좋은 실험이 매우 명확하다

최소한 다음 세 가지가 필요합니다.

### A. Natural-negative control

BEAF의 inpainted negative와 별도로

* 자연스럽게 object가 없는 image
* 동일 concept가 존재하는 image

를 구성해야 합니다.

그리고 같은 probe가

$$
\text{inpainted absent}
$$

뿐 아니라

$$
\text{naturally absent}
$$

도 구분하는지 확인해야 합니다.

이것이 없으면 image-side claim은 약합니다.

---

### B. Non-negation compositional control

이게 가장 중요합니다.

동일한 \(W\)를 사용하여

* negation
* attribute binding
* spatial relation
* object binding
* Winoground-style relation swap

을 모두 평가해야 합니다.

예를 들어:

| Scoring   | Negation | Attribute | Spatial | Winoground |
| --------- | -------: | --------: | ------: | ---------: |
| cosine    |      4.0 |         ? |       ? |          ? |
| rank-2 W  |        ? |         ? |       ? |          ? |
| rank-8 W  |        ? |         ? |       ? |          ? |
| rank-32 W |     29.2 |         ? |       ? |          ? |

만약 \(W\)가 negation에서만 크게 좋아진다면 논문의 주장이 훨씬 강해집니다.

반대로 모든 compositional task에서 좋아진다면 논문의 conclusion은

> "negation is a special similarity-interface failure"

가 아니라

> **"identity cosine is a poor compositional interaction function"**

으로 바뀌어야 합니다.

그리고 후자의 경우 오히려 `Is CLIP ideal?` 및 Winoground 계열과의 관계가 핵심이 됩니다. `Is CLIP ideal?`은 이미 CLIP cosine geometry 자체가 attribute binding, spatial relations, negation 등을 동시에 처리하는 데 구조적 한계가 있다고 주장합니다. 

---

# 8. 또 하나의 문제: rank-32 W는 "작은 구조"라고 하기 어렵다

논문은 full \(W\)가 262,144 parameters이고 rank-32가 그것보다 훨씬 작기 때문에

> "이득은 자유도가 아니라 구조에서 온다"

고 주장합니다. 

하지만 이것도 조금 과장되어 있습니다.

rank-32 matrix는

$$
W=UV^T
$$

라면 대략

$$
2d r
$$

개의 자유도를 가집니다.

CLIP ViT-B/32의 embedding dimension을 512라고 하면 대략

$$
2\times512\times32
=32768
$$

parameter입니다.

즉 full matrix 262,144보다는 작지만 **32K parameter는 결코 trivial하지 않습니다.**

42 concepts를 대상으로 학습하면서 32K parameters를 사용하는 상황에서 "작은 구조"라고 부르려면 훨씬 강한 control이 필요합니다.

특히

* rank 1
* rank 2
* rank 4
* rank 8
* rank 16
* rank 32
* random rank-r
* learned diagonal
* learned orthogonal
* low-rank + regularization

을 동일 조건에서 비교해야 합니다.

현재의 "full rank보다 rank-32가 좋다"는 사실만으로 구조적 원인을 입증했다고 보기는 어렵습니다.

---

# 9. 더 큰 문제: 29.19% 자체가 높지 않다

이 부분은 논문의 framing과 관계없이 냉정하게 봐야 합니다.

random group score는 16.67%이고, rank-32가 29.19%입니다. Winoground의 group score도 같은 엄격한 2×2 matching 관점에서 정의됩니다. Winoground에서는 당시 CLIP이 8% group score였고, human은 85.5%였습니다. 

따라서

$$
4.03\rightarrow29.19
$$

는 상대적으로는 7.2배지만,

$$
16.67\rightarrow29.19
$$

라고 보면 **random보다 약간 나은 수준**입니다.

즉 "7.2× improvement"는 인상적인 표현이지만 실제 capability 관점에서는 매우 제한적입니다.

논문이 진짜 말하고 싶은 것은

> cosine similarity가 정보를 제대로 읽지 못한다

이지

> 새로운 scoring function으로 negation understanding을 해결했다

가 아니어야 합니다.

이 구분이 중요합니다.

---

# 10. 4.03%라는 숫자도 주의해서 봐야 한다

논문은 4.03%가 random 16.67%보다 낮고, 심지어 cross-concept null의 0.67%보다 6배 높다고 강조합니다. 

수학적으로 흥미롭지만 리뷰어 입장에서는 이 비교가 약간 위험합니다.

왜냐하면

**0.67%는 random baseline이 아니라 저자들이 만든 특정 permutation/null construction의 결과**이기 때문입니다.

따라서 독자가 가장 먼저 보는 기준선은 여전히

$$
16.67\%
$$

입니다.

4.03%라는 값이 낮다는 것은 강력한 결과지만,

> "0.67%보다 6배 높다"

는 framing은 논문의 과학적 중요성을 실제보다 부풀려 보이게 할 위험이 있습니다.

저라면 본문에서는 오히려

> "The vanilla cosine score is systematically anti-compositional under this controlled 2×2 design, achieving only 4.03%, substantially below the 16.67% random group-score baseline."

처럼 쓰는 것이 낫다고 봅니다.

---

# 11. γ가 항상 양수라는 결과도 흥미롭지만 causal interpretation은 과하다

논문은 42 concepts 모두에서 \(\gamma>0\)이고 bootstrap CI가 positive라고 합니다. 

이것은 상당히 좋은 결과입니다.

왜냐하면

$$
\gamma\neq0
$$

라는 것은 최소한 image state와 text state 사이에 interaction이 존재한다는 것을 보여주기 때문입니다.

하지만

$$
\gamma>0
$$

라고 해서 그 interaction이 **negation understanding**이라고 바로 말할 수는 없습니다.

예를 들어 image와 text 모두에서 "cup"이라는 개념이 특정 semantic direction을 만들고, 그것이 positive/negative 상태와 상호작용하는 것만으로도 \(\gamma\)가 positive가 될 수 있습니다.

따라서 반드시 필요한 것은:

> \(\gamma\)가 실제 negation semantics에 특이적인가?

입니다.

이를 확인하는 가장 좋은 방법은 **polarity permutation control**입니다.

예를 들어 동일한 2×2 구조에서

* presence/absence
* positive/negative

label을 무작위로 재배열하거나

* object assignment를 바꾸거나
* negation operator만 제거하거나
* positive-positive / negative-negative caption pair를 사용하는 control

을 넣어 \(\gamma\)가 정말 negation interaction에 대응하는지 보여줘야 합니다.

---

# 12. 3.4 external task는 아직 "external validation"이라고 부르기 어렵다

COCO 5K에서 \(\beta\)와 negative-query performance drop의 correlation이

$$
r=0.835,\quad p=0.005
$$

라고 합니다. 

흥미로운 분석이기는 합니다.

하지만 표본이 **9 models**입니다.

$$
n=9
$$

에서

$$
r=0.835
$$

는 흥미롭지만, "external validation"이라고 강하게 부르기에는 표본이 너무 작습니다.

더 중요한 것은 이 correlation이

> negation performance

가 아니라

> affirmative retrieval 대비 performance drop

을 설명한다는 것입니다.

논문도 이것을 정확히 인정합니다. \(\gamma/|\beta|\)는 actual negation performance와 상관되지 않고, \(\beta\)가 performance drop을 예측한다고 합니다. 

따라서 이것은 논문의 핵심 claim을 검증한다기보다는 **분해의 descriptive validity를 보여주는 auxiliary experiment**에 가깝습니다.

---

# 13. 기존 연구 대비 novelty는 "있지만, 매우 좁다"

이 부분을 냉정하게 평가하면:

### 이미 알려진 것

* CLIP은 negation에 약하다 — CVPR 2025 NegBench 
* negation signal이 representation에 존재할 수 있다 — ACL 2024 및 후속 연구 
* intermediate layer에서 negation signal을 찾을 수 있다 — PeakPatch 
* embedding geometry에 negation direction이 존재한다 — CVPR 2026 
* similarity function 자체가 compositional reasoning의 병목이 될 수 있다 — `Is CLIP ideal?` 
* similarity를 논리적 추론 대신 사용하면 문제가 생긴다는 방향 — 논문의 related work가 인용한 Similarity Is Not Logic
* identical-word-set pair를 사용하여 compositional matching을 검증한다 — Winoground 

### 이 논문이 새롭게 추가하는 것

**"representation-level signal의 존재와 vanilla cross-modal decision에서 그 signal이 충분한 크기로 나타나는 것은 별개의 문제이며, 이를 2×2 factorial interaction \(\gamma\)로 정량화할 수 있다."**

이것은 실제로 novel한 framing입니다.

다만 "새로운 method"라기보다는 **새로운 empirical diagnostic**입니다.

따라서 top-tier conference에서 accept되려면 **diagnostic이 정말 새로운 scientific understanding을 만들어냈다는 증거**가 필요합니다.

현재는 그 정도까지는 못 갔습니다.

---

# 14. 제가 리뷰어라면 다음과 같이 점수를 줄 것 같습니다

대략 다음과 같습니다.

| 항목                    |          평가 |
| --------------------- | ----------: |
| 문제 중요성                |        8/10 |
| 아이디어 명확성              |        8/10 |
| 수학적 정확성               |        8/10 |
| 실험 설계                 |        5/10 |
| novelty               |        6/10 |
| 기존 연구와의 차별성           |        6/10 |
| empirical evidence    |        5/10 |
| generalization        |        3/10 |
| claim의 강도 대비 evidence |        4/10 |
| 전체                    | **5/10 전후** |

그리고 recommendation은

**Reject / Weak Reject**

입니다.

---

# 15. 그러나 이 논문을 살릴 방법은 상당히 명확하다

저라면 논문을 완전히 버리지는 않습니다.

오히려 주장 자체를 조금 낮추고 실험을 정확히 강화하겠습니다.

현재 논문의 위험한 claim:

> **The bottleneck is not representation but the similarity interface.**

이것은 너무 강합니다.

더 방어 가능한 claim:

> **Negation information is linearly decodable from CLIP representations, yet its cross-modal interaction is too weak relative to unimodal effects to determine standard cosine-based matching decisions.**

이렇게 바꾸면 훨씬 강합니다.

그리고 핵심 실험은 딱 세 개면 됩니다.

### 실험 1 — Natural absence

BEAF inpainting artifact가 아닌 자연 이미지에서도 동일한 결과가 나오는가?

### 실험 2 — Compositional control

동일한 \(W\)를

* negation
* attribute binding
* spatial relation
* Winoground

에 적용합니다.

이 실험이 논문의 novelty를 결정합니다.

### 실험 3 — Metric-specificity

$$
S=v^TWt
$$

가 단순히 general alignment를 개선한 것이 아니라는 것을 보여줍니다.

예를 들어:

$$
\text{cosine}
\rightarrow
\text{diagonal}
\rightarrow
\text{low-rank W}
\rightarrow
\text{full W}
$$

를 비교하고, **held-out concept**에서 평가해야 합니다.

---

# 16. 특히 Winoground를 직접 연결하면 논문의 위치가 훨씬 좋아진다

현재 논문은 Winoground를 사실상 "우리의 2×2 metric과 동일하다"는 수준으로 사용합니다.

저는 이보다 훨씬 적극적으로 연결해야 한다고 봅니다.

Winoground의 핵심 발견은 **단어가 동일해도 compositional arrangement가 다르면 CLIP이 이를 제대로 구분하지 못한다**는 것이었습니다. CLIP의 Winoground group score는 8%였고, human은 85.5%였습니다. 

현재 논문은 한 단계 더 나아가 다음 질문을 할 수 있습니다.

> Winoground-type failure도
>
> $$
> \gamma < \max(|\alpha|,|\beta|)
> $$
>
> 형태로 설명되는가?

만약 그렇다면 이것은 단순한 "negation paper"가 아니라,

**"CLIP compositional failure를 main effect vs interaction의 부족으로 통합적으로 설명하는 framework"**

가 됩니다.

이건 훨씬 큰 논문입니다.

반대로 negation에서만 이 현상이 나타난다면,

**"negation은 다른 compositional phenomena와 달리 interaction signal이 특히 약하다."**

라는 새로운 발견이 됩니다.

어느 쪽이든 지금보다 훨씬 강합니다.

---

# 최종 판정

현재 제출본에 대해서는 **Reject**입니다.

하지만 이유는 "아이디어가 별로라서"가 아닙니다.

오히려 문제는 **아이디어가 주장하는 것보다 실험이 아직 약하다**는 것입니다.

제가 가장 중요하게 보는 세 가지 결함은 다음입니다.

1. **Linear probe ≠ cross-modal usability**인데 둘 사이의 논리적 간극을 충분히 메우지 못했습니다.
2. **rank-32 bilinear scoring의 7.2× 개선이 negation-specific인지 입증하지 못했습니다.**
3. **BEAF inpainting이 만들어내는 visual artifact를 absence representation과 분리하지 못했습니다.**

반대로 가장 가치 있는 부분은 분명합니다.

$$
\boxed{
\text{representation contains information}
\neq
\text{similarity can use the information}
}
$$

이라는 구분을 **2×2 interaction으로 정량화했다는 것**입니다. 이 framing 자체는 기존 NegBench, ACL 2024 분석, CVPR 2026 geometry 연구, PeakPatch, 그리고 Winoground와 상당히 잘 맞물립니다.    

따라서 현재 상태를 한 문장으로 평가하면:

> **"좋은 논문이 될 수 있는 진단적 핵심은 발견했지만, 현재 실험만으로는 'similarity interface가 negation의 본질적 bottleneck이다'라는 결론을 받아들이기 어렵다."**

특히 **Winoground-style compositional control + natural absence control + W의 negation-specificity 검증** 세 가지가 추가된다면, 단순한 2-page observation에서 상당히 설득력 있는 논문으로 올라갈 가능성이 있습니다.

리뷰어 관점에서 엄격하게 평가하겠습니다. 결론부터 말하면, **현재 원고는 아이디어 자체는 흥미롭지만 국내 학술대회에서 “양질의 논문만 accept한다”는 기준을 적용하면 저는 Reject에 가깝게 판단합니다.** 다만 이는 연구 방향이 나쁘다는 의미가 아니라, 현재 2페이지 원고가 주장하는 수준과 실험이 실제로 입증하는 수준 사이에 아직 중요한 간극이 있기 때문입니다.

특히 이 원고는 Winoground의 핵심 아이디어인 “표면적으로 동일한 구성요소를 유지한 채 관계만 바꾸어 모델의 진짜 compositional reasoning을 측정한다”는 발상과 상당히 잘 맞습니다. Winoground 역시 단어 집합을 동일하게 유지하면서 순서만 바꾼 caption pair와 대응하는 image pair를 구성하여 단순한 단어 존재 여부만으로는 해결할 수 없도록 했습니다. 

### 종합 판정

**판정: Reject / Weak Reject**

**신뢰도: 높음**

이유는 간단합니다.

> 이 논문의 가장 좋은 아이디어는 “부정 정보가 표현에 존재하는가?”와 “그 정보가 실제 similarity ranking을 결정할 수 있는가?”를 분리하는 것입니다.

이것은 기존 연구와 단순히 같은 말을 반복하는 것은 아닙니다. 그러나 현재 실험은 그 다음 단계인

> “따라서 CLIP의 핵심 병목은 visual presence/absence representation이다.”

까지는 충분히 입증하지 못합니다.

오히려 현재 결과만으로는 더 보수적인 결론인

> “현재 구성한 counterfactual benchmark에서 CLIP의 최종 cosine similarity는 객체 존재/부재의 cross-modal interaction을 충분히 활용하지 못한다.”

정도가 안전합니다.

이 차이가 상당히 큽니다.

---

## 1. 가장 좋은 부분: 문제를 2×2 factorial structure로 재구성한 것

논문의 핵심 분석은 상당히 깔끔합니다.

이미지 상태 \(a\in\{+1,-1\}\), 텍스트 상태 \(b\in\{+1,-1\}\)에 대해

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

로 표현하고,

* \(\alpha\): text main effect
* \(\beta\): image main effect
* \(\gamma\): image-text interaction

으로 해석하는 구조는 논리적으로 타당합니다.

특히

$$
\Delta
=
\min(S_{++},S_{--})
-
\max(S_{+-},S_{-+})
$$

에 대해

$$
\Delta
=
2\gamma-2\max(|\alpha|,|\beta|)
$$

를 얻고,

$$
\gamma>\max(|\alpha|,|\beta|)
$$

일 때에만 2×2 matching이 성공한다는 분석은 이 논문의 가장 강한 부분입니다. 

이 점은 Winoground와의 연결도 적절합니다. Winoground 역시 두 이미지와 두 caption의 네 가지 matching 중 올바른 두 diagonal pairing을 동시에 선택해야 하는 group score를 사용합니다. 실제 Winoground에서 CLIP ViT-B/32의 group score가 8%였다는 점도 이 문제 설정의 어려움을 보여줍니다. 

따라서 **수학적 decomposition 자체는 좋은 분석 도구**입니다.

문제는 이것이 곧바로 새로운 scientific finding이 되는 것은 아니라는 점입니다.

---

# 2. Novelty: “있지만, 생각보다 약하다”

기존 연구를 모두 고려하면 novelty를 과장해서는 안 됩니다.

이미 다음과 같은 방향들이 존재합니다.

* NegBench: CLIP이 negation에서 실패한다는 대규모 benchmark와 fine-tuning 분석 
* Quantmeyer et al.: CLIP 내부에서 negation 정보가 처리되는 위치를 causal tracing과 attention analysis로 분석 
* Sammani et al.: CLIP embedding space에 negation direction이 존재하며 representation intervention으로 이를 조작할 수 있다고 주장 
* SpaceVLM: negation을 point가 아니라 subspace로 모델링 
* Aggarwal et al.: text embedding의 semantic information을 직접 제거하는 방식 
* 최근 PeakPatch: intermediate representation에 negation signal이 존재하며 final representation에서 collapse된다는 분석 

따라서 “CLIP의 representation에는 negation 정보가 있고 cosine similarity가 이를 제대로 활용하지 못한다”는 것 자체는 이제 새로운 주장이 아닙니다.

그런데 이 논문이 제시하는 차별점은 꽤 명확합니다.

**기존 연구가 representation의 존재 여부와 similarity의 실패를 서로 다른 실험에서 보여줬다면, 이 논문은 동일한 2×2 counterfactual block에서 두 현상을 하나의 수학적 decomposition으로 연결한다.**

이것은 novelty로 인정할 수 있습니다.

특히 논문의 다음 결과는 흥미롭습니다.

$$
|\alpha|=0.00380,\quad
|\beta|=0.00628,\quad
\gamma=0.00121
$$

즉 interaction은 존재하지만 main effect보다 훨씬 작으며, 실제 2×2 matching accuracy는 4.03%에 불과합니다. 

그리고 cross-concept pairing을 이용한 control에서 0.67%가 나온다는 결과도 단순 chance 16.67%만 사용하는 것보다 훨씬 설득력이 있습니다. 

**따라서 “새로운 분석 프레임워크”로서는 충분히 흥미롭습니다.**

다만 “새로운 원인 규명”으로 주장하기에는 아직 부족합니다.

---

# 3. 가장 큰 문제: 이미지 AUC 0.759가 무엇을 의미하는가?

제가 리뷰어라면 가장 먼저 이 부분을 공격하겠습니다.

논문은

> linear probe로 이미지에서 객체 존재/부재 정보가 검출된다.

고 주장합니다.

결과는 macro AUC 0.759입니다. 

하지만 이것은 상당히 약한 증거입니다.

왜냐하면 이미지가

$$
I_{\text{original}}
\quad\text{vs.}\quad
I_{\text{inpainted}}
$$

이기 때문입니다.

즉 probe가 실제로

> “고양이가 존재한다/존재하지 않는다”

를 검출하는 것인지,

아니면

> “이 이미지가 BEAF의 before/after/inpainting 이미지처럼 생겼다”

를 검출하는 것인지

구별되지 않습니다.

논문 스스로도 이 문제를 인정합니다. 반사실 이미지는 inpainting으로 생성되었고, 원래부터 해당 객체가 없는 자연 이미지와 비교하지 않았다고 명시합니다. 

이것은 단순한 limitation이 아닙니다.

**논문의 핵심 주장에 직접 영향을 주는 confound입니다.**

예를 들어 probe가 inpainting artifact, texture discontinuity, object boundary, local blur 등을 이용한다면 AUC 0.759는 “CLIP이 객체 부재를 representation에 저장한다”는 증거가 아닙니다.

오히려 “CLIP이 두 종류의 이미지 변형을 구별한다”는 증거일 수 있습니다.

이 문제를 해결하지 않은 상태에서

> “CLIP은 객체가 사라졌다는 사실을 알아채지 못한다.”

라고 결론내리는 것은 과도합니다.

---

# 4. 두 번째 치명적 문제: “정보가 있다”의 operational definition이 너무 약하다

텍스트 쪽은 더욱 주의해야 합니다.

AB-swap을 사용한 것은 좋은 선택입니다.

예를 들어

> pizza + no cup

과

> cup + no pizza

처럼 같은 token multiset을 유지하기 때문에 단순 bag-of-words 전략으로 해결하기 어렵습니다.

이 점은 Winoground의 철학과 정확히 맞닿아 있습니다. Winoground 역시 caption의 단어 집합을 동일하게 유지하여 단어 자체의 존재가 아닌 compositional arrangement를 요구합니다. 

논문은 텍스트 probe accuracy가 73.80%이고, 위치 leakage를 통제하면 64.69%까지 떨어지지만 여전히 chance 25%보다 높다고 보고합니다. 

이것은 **정보가 있다는 약한 증거**입니다.

하지만 여기에도 문제가 있습니다.

64.69%가 정확히 무엇을 의미하는지 더 엄격한 control이 필요합니다.

특히:

1. token position
2. sentence length
3. punctuation
4. negation token 자체의 위치
5. template
6. lexical frequency
7. concept-specific frequency

등이 충분히 통제되었는지가 2페이지에서는 불분명합니다.

“네 개 template 계열을 가로지르는 전이가 95% 유지된다”는 것은 좋은 결과지만, 이것만으로 compositional semantics가 encoding되었다고 단정하기에는 부족합니다.

---

# 5. 가장 중요한 논리적 약점: probe → retrieval 사이의 연결

논문의 핵심 문제의식은 정확합니다.

> linear probe에서 정보가 검출된다 ≠ cosine similarity가 그 정보를 사용할 수 있다.

이것은 좋은 지적입니다.

그러나 논문은 반대로도 너무 빠르게 갑니다.

> cosine similarity에서 실패한다 → information representation 자체가 병목이다.

이것 역시 자동으로 성립하지 않습니다.

왜냐하면 현재 실험에서 보여준 것은

$$
\text{probe information}
$$

과

$$
\text{cosine interaction}
$$

의 차이이지,

$$
\text{visual representation itself is insufficient}
$$

의 증명이 아니기 때문입니다.

특히 이미 다른 연구에서는 CLIP intermediate representation에서 negation-related information이 관찰되고, final embedding에서 그 신호가 약화된다는 결과가 있습니다. Quantmeyer et al.은 CLIP text encoder에서 negation 관련 attention과 layer-localized processing을 보고했고 , 최근 PeakPatch 역시 intermediate layer에서 compositional signal을 복구하는 접근을 취합니다. 

따라서 이 논문이 정말 증명해야 하는 것은

> **“visual encoder가 absence를 제대로 표현하지 못한다.”**

입니다.

현재는

> **“final joint embedding의 cosine interface가 absence information을 제대로 활용하지 못한다.”**

까지만 강하게 말할 수 있습니다.

이 둘은 상당히 다릅니다.

---

# 6. rank-32 linear transformation 실험은 흥미롭지만 해석이 위험하다

이 부분은 상당히 좋은 실험입니다.

동일한 embedding을

$$
S_W=v^\top Wt
$$

로 다시 채점하고, 개념 단위 GroupKFold를 적용하여 학습에 등장하지 않은 concept에서 평가했다는 점은 좋습니다. rank-32에서 29.19%까지 올라가며 cosine의 7.2배라는 결과도 흥미롭습니다. 

또한

* random rank-1: 0.44%
* shuffled labels: 0.69%
* full \(W\): 23.91%
* rank-32: 29.19%

라는 비교는 단순히 parameter 수가 많아서 생긴 결과가 아니라는 주장을 어느 정도 뒷받침합니다. 

그러나 여기서 논문의 표현:

> “병목은 표현도 아니고 필요한 구조의 복잡도도 아니다.”

는 **너무 강합니다.**

rank-32 \(W\)가 성능을 개선했다는 것은 단지

> 현재 final embedding에 어느 정도 usable signal이 존재한다.

는 것을 보여줍니다.

그것이

> representation이 충분하다.

는 것을 의미하지는 않습니다.

29.19%는 여전히 매우 낮습니다. 게다가 16.67%가 naive 2×2 chance이기는 하지만, 논문 자체의 cross-concept null이 0.67%라는 특수한 값이므로 어떤 baseline을 “진짜 chance”로 사용할 것인지 더 엄밀한 설명이 필요합니다.

더 중요한 것은 rank-32가 **29.19%밖에 못 올린다는 사실 자체가 오히려 representation의 한계를 남겨둡니다.**

즉 이 실험은 논문의 주장보다 더 보수적으로 해석하는 것이 맞습니다.

> “cosine은 information을 충분히 읽어내지 못하지만, linear bilinear scoring으로 일부 interaction을 회수할 수 있다.”

여기까지입니다.

---

# 7. “7.2배”는 리뷰어에게 크게 어필하지 않을 가능성이 높다

논문의 29.19% / 4.03% = 약 7.2배라는 표현은 인상적입니다.

그러나 absolute performance는

$$
4.03\%\rightarrow29.19\%
$$

입니다.

그리고 2×2 group task의 random chance가 16.67%이므로, 29.19%가 상당히 의미 있는 개선인 것은 맞지만 “해결했다”고 보기 어렵습니다.

Winoground에서도 group score를 핵심 지표로 사용하면서 모델이 chance 부근 또는 그 이하에 머무는 것을 compositional failure의 중요한 증거로 삼았습니다. 

따라서 이 논문도 29.19%를 “cosine보다 훨씬 낫다”라고 표현하는 것은 괜찮지만,

> representation is sufficient

혹은

> cosine is the sole bottleneck

으로 확장하면 과장입니다.

---

# 8. γ 자체의 통계적 유의성은 좋은데, effect size 문제가 남는다

논문은 상당히 좋은 방어를 하고 있습니다.

$$
\gamma=0.00121
$$

이지만 42개 concept 전체에서 bootstrap CI가 0보다 크고, macro CI도

$$
[0.00111,0.00130]
$$

이며, 모든 42개 concept에서 positive라고 보고합니다. 

따라서

> “γ가 사실상 noise다.”

라는 공격은 상당 부분 방어됩니다.

하지만 scientific importance는 별개입니다.

논문 스스로 보여주는 것처럼

$$
\frac{\gamma}{\max(|\alpha|,|\beta|)}
=0.1828
$$

수준입니다.

즉 **통계적으로 존재하는 interaction과 실제 decision을 뒤집을 만큼 강한 interaction은 완전히 다른 문제**입니다.

오히려 이 부분은 논문의 좋은 메시지입니다.

따라서 “γ가 유의하다”보다

> **“interaction은 존재하지만 decision-relevant scale까지 성장하지 못한다.”**

를 핵심 메시지로 잡는 것이 더 강합니다.

---

# 9. 가장 큰 실험적 결함: 자연 이미지 검증 부재

저라면 이 문제 하나만으로도 major revision을 요구합니다.

현재 모든 핵심 분석이 BEAF의 counterfactual image에 의존합니다.

BEAF는 동일 장면에서 특정 객체를 제거한 반사실 쌍이라는 장점이 있습니다. 하지만 그 장점 자체가 confound가 됩니다.

현재 논문의 질문은

> “객체의 존재/부재를 representation이 encode하는가?”

인데,

실험은 사실상

> “원본 이미지와 inpainted image를 representation이 구분하는가?”

입니다.

이것은 동일하지 않습니다.

최소한 다음 세 조건이 필요합니다.

$$
I_{\text{with}}
\leftrightarrow
I_{\text{without-natural}}
$$

$$
I_{\text{with}}
\leftrightarrow
I_{\text{without-inpaint}}
$$

$$
I_{\text{without-inpaint}}
\leftrightarrow
I_{\text{without-natural}}
$$

를 비교해야 합니다.

특히 “같은 장면에서 객체만 제거”라는 핵심 장점을 유지하면서도 **inpainting artifact에 대한 control**이 있어야 합니다.

---

# 10. “이미지 쪽이 병목이다”라는 결론은 아직 premature

이 논문의 가장 위험한 문장은 사실 여기입니다.

현재 데이터는

* image probe: AUC 0.759
* text probe: 64.69%
* cosine group accuracy: 4.03%

입니다. 

이것만 보면 visual side가 더 약해 보일 수 있습니다.

하지만 modality 간 probe accuracy를 직접 비교하는 것은 적절하지 않습니다.

AUC와 accuracy는 서로 다른 통계량이고, 각각의 label structure도 다릅니다.

더구나 논문의 2×2 decomposition에서는

$$
\gamma
\approx
\frac14
\|d_I\|
\|d_T\|
\cos(d_I,d_T)
$$

이고,

$$
\cos(d_I,d_T)=0.167
$$

이라고 보고합니다. 

이 결과는 오히려 상당히 흥미로운 다른 해석을 가능하게 합니다.

> visual magnitude가 작아서가 아니라, **visual polarity direction과 textual polarity direction의 cross-modal alignment가 약해서 interaction이 작다.**

즉 “visual representation failure”와 “cross-modal alignment failure”가 아직 분리되지 않았습니다.

이 점에서 제목과 결론을 조심해야 합니다.

---

# 11. 기존 논문들과의 관계는 오히려 잘 잡을 수 있다

흥미롭게도 이 논문은 기존 연구와 경쟁하기보다 하나의 통합적인 분석 틀을 제공할 가능성이 있습니다.

예를 들어 기존 연구의 주장을 현재 notation으로 보면,

* “negation information이 representation에 있다” → \(\gamma\neq0\)
* “main concept similarity가 negation을 압도한다” → \(|\alpha|,|\beta|\gg\gamma\)
* “scoring rule을 바꾸면 좋아진다” → \(W\neq I\)
* “negation direction을 steering하면 된다” → \(d_T\) 또는 관련 representation을 수정
* “fine-tuning으로 개선된다” → \(\alpha,\beta,\gamma\)의 상대적 구조를 학습 과정에서 변화

처럼 표현할 수 있습니다.

이렇게 되면 논문의 contribution은 “새로운 negation method”가 아니라

> **기존 negation 연구를 하나의 interaction framework로 재해석하고, representation-level evidence와 decision-level success를 분리한다.**

가 됩니다.

이것은 꽤 괜찮은 논문 포지션입니다.

---

# 12. 하지만 현재 2페이지 논문은 너무 많은 것을 주장한다

현재 원고의 결론은 사실 세 가지 논문을 한꺼번에 주장합니다.

첫째,

> negation information은 representation에 존재한다.

둘째,

> cosine interface는 이를 활용하지 못한다.

셋째,

> 근본적인 병목은 visual absence representation이다.

첫째는 어느 정도 입증됩니다.

둘째는 상당히 잘 입증됩니다.

셋째가 문제입니다.

**저라면 셋째를 현재 논문에서 삭제하거나 상당히 약화시키겠습니다.**

현재 증거로 가장 방어하기 쉬운 논문은:

> “Object-Presence Negation in Multi-Object Scenes: Measuring the Gap Between Represented Information and Similarity Interaction”

입니다.

반면

> “Negation의 병목은 text가 아니라 vision이다”

까지 가려면 추가 실험이 필요합니다.

---

# 13. 최종 리뷰 의견

**장점**

이 논문의 가장 큰 장점은 단순한 “CLIP은 negation을 못한다”는 재현이 아니라는 점입니다. 2×2 counterfactual design을 이용하여 image main effect, text main effect, cross-modal interaction을 명시적으로 분리하고, 최종 ranking 성공 조건을

$$
\gamma>\max(|\alpha|,|\beta|)
$$

라는 하나의 조건으로 정리한 것은 깔끔합니다. 또한 AB-swap은 Winoground의 compositional stress-test 철학과 잘 맞으며, 단순 token presence를 이용한 shortcut을 억제하려는 설계도 적절합니다. 

**그러나 결정적인 약점**

가장 중요한 실험적 문제는 visual probe가 실제 객체 부재가 아니라 BEAF inpainting artifact를 학습했을 가능성을 제거하지 못했다는 것입니다. 논문도 자연적으로 객체가 없는 이미지와의 비교를 하지 않았음을 인정합니다. 

또한 linear probe의 검출 가능성을 “representation이 충분하다”로 해석해서는 안 됩니다. rank-32 bilinear scoring에서 성능이 29.19%까지 올라간 것은 final embedding에 interaction-relevant information이 있음을 보여주지만, 동시에 대부분의 matching이 여전히 실패한다는 사실도 보여줍니다. 

따라서 현재 원고의 핵심 conclusion인 **“병목은 representation도 아니고 구조의 복잡도도 아니다”** 또는 **“CLIP은 객체가 사라졌다는 사실을 알아채지 못한다”**는 실험 결과보다 한 단계 앞서 나간 주장입니다.

### 최종 점수

| 항목           |      평가 |
| ------------ | ------: |
| 문제 중요성       |     4/5 |
| 아이디어/분석 독창성  |     4/5 |
| 이론적 명료성      |     4/5 |
| 실험 설계        |     3/5 |
| 실험적 설득력      |   2.5/5 |
| 기존 연구 대비 차별성 |   3.5/5 |
| 결론의 타당성      |   2.5/5 |
| 재현성/검증 가능성   |     3/5 |
| **종합**       | **3/5** |

**Recommendation: Reject (borderline Weak Reject)**

다만 저는 이 논문을 “아이디어가 약해서 Reject”하지는 않습니다. **핵심 분석 자체는 국내 학술대회에서 충분히 발표할 만한 수준입니다. 문제는 가장 중요한 causal interpretation을 아직 증명하지 못했다는 것입니다.**

리뷰어 입장에서 단 하나의 추가 실험만 요구한다면, 저는 **“inpainting artifact를 제거한 자연적 absence control”**을 요구하겠습니다. 그리고 그 결과가 유지된다면 평가를 상당히 올릴 수 있습니다.

반대로 그 control에서 이미지 AUC가 크게 무너지면, 현재 논문의 가장 중요한 주장인 **“visual absence representation 자체가 병목”**은 철회해야 합니다.

따라서 지금 단계에서 가장 현실적인 평가는 **“흥미롭고 잘 구성된 진단 논문의 씨앗이지만, 핵심 주장에 대한 결정적 control이 빠져 있어 accept하기에는 이르다”**입니다.
리뷰어 관점에서 상당히 엄격하게 평가하겠습니다. 결론부터 말하면, **현재 원고는 흥미로운 진단 결과를 갖고 있지만, 현 상태라면 저는 Accept보다는 Weak Reject 쪽에 투표하겠습니다.** 다만 이는 아이디어가 약해서가 아니라, 핵심 주장에 비해 실험적·논리적 방어가 아직 충분하지 않기 때문입니다. 특히 2페이지 학부논문경진대회라는 조건에서는 “복잡한 분석을 많이 했다”보다 “하나의 주장이 빈틈없이 입증되었는가”가 더 중요합니다.

### 1. 제가 이해한 논문의 핵심 주장

현재 논문의 가장 좋은 부분은 문제를 단순히 “CLIP이 negation을 이해하지 못한다”로 반복하지 않는다는 것입니다.

논문의 주장은 대략 다음입니다.

> CLIP의 이미지·텍스트 표현에는 객체의 존재/부재 및 문장의 극성에 관한 정보가 어느 정도 존재하지만, 두 정보가 최종 cosine similarity에서 충분히 강하게 상호작용하지 못하기 때문에 negation matching에 실패한다.

이를 2×2 요인설계로 표현하여

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

로 분해하고, 여기서 \(\gamma\)를 이미지 상태와 텍스트 극성의 상호작용으로 해석합니다. 그리고

$$
\Delta
=
\min(S_{++},S_{--})
-
\max(S_{+-},S_{-+})
=
2\gamma-2\max(|\alpha|,|\beta|)
$$

이므로 정확한 matching에는

$$
\gamma>\max(|\alpha|,|\beta|)
$$

가 필요하다는 것이 논문의 핵심입니다. 

이 아이디어 자체는 상당히 깔끔합니다. 특히 “정보가 존재한다”와 “그 정보가 cosine ranking을 결정한다”를 분리한 것은 기존 negation 연구와 차별화될 수 있는 좋은 framing입니다.

기존 연구들이 실제로 서로 다른 위치를 공격하고 있다는 점도 논문의 문제의식과 잘 맞습니다. NegBench는 CLIP의 negation failure 자체를 대규모로 보여주었고,  Sammani et al.은 negation direction의 존재와 steering을 연구했으며,  Koishigarina et al. 계열의 연구는 unimodal representation에 관련 정보가 존재할 수 있음을 보입니다. 반대로 최근의 PeakPatch 역시 “중간 layer에는 negation signal이 있으나 최종 표현에서 사라진다”는 방향으로 주장합니다. 

따라서 **“negation information이 있는가?”에서 “그 정보가 cross-modal decision을 지배할 정도로 강한가?”로 질문을 옮기는 것**은 충분히 논문이 될 수 있습니다.

문제는 그 다음입니다.

---

## 2. 가장 큰 문제: 2×2 분해 자체는 contribution이 아니다

논문도 이를 인정하고 있습니다. 2×2 분해는 표준적인 요인분해이고 contribution은 실제 측정값이라고 명시합니다. 

그런데 리뷰어 입장에서는 여기서 상당히 까다로운 질문을 하게 됩니다.

**“그러면 실제 측정값이 기존 연구에서 알려지지 않았던 사실인가?”**

현재 원고는 이 질문에 충분히 강하게 답하지 못합니다.

Winoground가 이미 동일한 단어 집합을 유지하면서 단어 순서와 시각적 관계의 결합을 테스트했습니다. 핵심 아이디어 역시 “unimodal information이 있는 것”과 “cross-modal composition을 제대로 수행하는 것”을 구별하는 데 있습니다. Winoground의 group score는 두 이미지와 두 caption을 올바르게 함께 매칭해야 하며, random chance는 16.67%입니다. 실제 CLIP의 group score도 8%로 매우 낮았습니다. 

따라서 현재 논문의

> “두 modality의 정보는 어느 정도 존재하지만 cross-modal matching에서는 실패한다”

라는 큰 방향 자체는 Winoground의 철학과 상당히 유사합니다.

다만 여기에는 중요한 차이가 있습니다.

Winoground는 **compositionality 전반**을 테스트합니다. 논문은 그 구조를 **object presence × textual polarity라는 특정한 negation 문제에 대해 2×2 causal-looking decomposition으로 구체화**합니다. 이것은 차별점이 될 수 있습니다.

하지만 현재 원고는 이 차이를 충분히 설명하지 않습니다.

리뷰어가 읽으면:

> “Winoground의 2×2 matching을 negation-specific하게 다시 만든 것 아닌가?”

라는 의문이 생깁니다.

따라서 Introduction에서 반드시 다음을 명확히 해야 합니다.

**Winoground가 보여준 것은 “composition이 실패한다”이고, 본 연구가 보여주는 것은 “negation에서 composition failure의 원인이 interaction term의 상대적 크기 부족으로 정량화된다”는 것**이라고 구분해야 합니다.

현재는 이 논리적 bridge가 너무 짧습니다.

---

# 3. 가장 심각한 약점: \(\gamma\)의 해석이 생각보다 강하지 않다

여기가 제가 리뷰어라면 가장 먼저 공격할 부분입니다.

논문은

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

에서 \(\gamma\)를 cross-modal interaction으로 해석합니다.

수학적으로는 맞습니다.

하지만 **\(\gamma\neq0\)이라는 사실이 곧 “의미적 cross-modal interaction 정보가 존재한다”는 뜻은 아닙니다.**

왜냐하면 \(\gamma\)는 네 개의 similarity 값에 대한 algebraic interaction coefficient일 뿐이기 때문입니다.

논문에서도 실제로 \(\gamma>0\)임을 관찰하고, 42개 concept 모두에서 bootstrap CI가 0보다 크다고 보고합니다. 

그러나 리뷰어는 이렇게 물을 수 있습니다.

> 이 positive \(\gamma\)가 정말 “pizza의 존재 × pizza의 부정”이라는 의미적 interaction인가? 아니면 CLIP embedding geometry, background correlation, image editing artifact, caption template 구조에서 발생하는 단순한 statistical interaction인가?

논문은 객체 불일치 permutation 결과를 제시하여 이를 방어하려고 합니다. 이것은 좋은 시도입니다.

하지만 **2페이지 논문에서 가장 중요한 검증이라면 본문에서 훨씬 더 강하게 보여줘야 합니다.**

현재는 “\(\gamma\)가 양수다 → interaction이 존재한다”로 너무 빠르게 넘어갑니다.

---

# 4. BEAF counterfactual의 문제가 매우 크다

이것은 Accept를 결정적으로 어렵게 만드는 요소입니다.

이미지 최소쌍은 BEAF의 원본 이미지와 특정 객체를 inpainting으로 제거한 이미지를 사용합니다. 논문도 이 점을 명시합니다. 

그런데 이 방식은 동시에 두 가지를 의미합니다.

1. 객체가 정말 사라졌는가?
2. **객체가 사라진 것 외에는 정말 아무것도 변하지 않았는가?**

후자는 보장되지 않습니다.

예를 들어 사람을 제거하면:

* 주변 texture가 바뀔 수 있고
* object boundary가 바뀌고
* shadow가 없어지고
* composition이 달라지고
* image statistics가 변합니다.

따라서 image encoder의 \(a\) 방향이 “object presence”가 아니라 “inpainting-induced change”를 잡고 있을 가능성이 있습니다.

논문도 이 한계를 인정합니다. 원래부터 해당 객체가 없는 자연 이미지와 비교하지 않았다고 명시합니다. 

하지만 이건 단순한 limitation이 아닙니다.

**논문의 핵심 전제 자체와 관련된 confound입니다.**

특히 논문의 결론이

> “image representation에는 object-presence information이 있다”

라면 이 문제는 치명적입니다.

현재 결과의 AUC 0.759가 정말 object presence를 측정하는지, 아니면 image-editing artifact를 측정하는지 분리되어야 합니다.

따라서 리뷰어라면 다음 실험을 요구하겠습니다.

* 자연적으로 해당 객체가 없는 이미지와 존재 이미지 비교
* object size에 따른 성능
* inpainting 영역의 pixel-level change와 probe score correlation
* 가능하면 다른 object-removal 방법과의 재현

이것이 없으면 저는 0.759라는 숫자를 논문의 강한 증거로 인정하기 어렵습니다.

---

# 5. 텍스트 probe도 완전히 안전하지 않다

텍스트 쪽은 훨씬 좋아 보입니다.

AB-swap을 사용해서

> “The scene features a pizza, but lacks a cup.”

와

> “The scene features a cup, but lacks a pizza.”

처럼 동일한 단어 집합을 유지하려 한 것은 적절합니다. 

또한 어순 shortcut을 검사하여 73.80%에서 64.69%로 감소하지만 여전히 chance보다 높다는 결과도 상당히 유용합니다. 

하지만 여기에도 문제가 있습니다.

**동일한 단어 multiset ≠ 동일한 linguistic information.**

두 문장의 syntactic structure가 완전히 동일한 것도 아니고, negated object의 위치가 완전히 균등하게 통제됐다는 보장도 없습니다.

논문은 위치 shortcut을 일부 검사하지만, 다음과 같은 가능성은 여전히 남습니다.

* 특정 object가 특정 template에서 자주 등장
* object-token의 위치 차이
* “lacks X”와 “features X”의 token interaction
* sentence length
* 특정 object의 lexical statistics

따라서 64.69%를 곧바로 “object-level negation representation”이라고 부르는 것은 조금 과합니다.

더 안전한 표현은:

> “the text representation contains information sufficient to discriminate the negated object beyond a tested word-order shortcut.”

정도입니다.

---

# 6. 4.03%라는 결과는 강력하지만, 해석이 상당히 위험하다

이 논문의 가장 눈에 띄는 숫자는 사실 4.03%입니다.

Random permutation의 theoretical chance는 16.67%이고, 논문의 4.03%는 그보다 훨씬 낮습니다. Winoground에서도 group score chance가 16.67%라는 동일한 구조를 사용합니다. 

따라서 이것은 흥미로운 결과입니다.

그런데 논문은

> “교차항이 존재하지만 주효과보다 작아서 실패한다”

라고 설명합니다.

여기까지는 상당히 설득력 있습니다.

하지만 **4.03%가 반드시 “CLIP이 거의 완전히 틀렸다”는 것을 의미하지는 않습니다.**

왜냐하면 2×2 task의 네 score가 강한 공통 scene similarity를 공유하기 때문입니다.

논문도 실제 데이터에서 cross-concept null이 0.67%라고 보고합니다. 

오히려 이 결과는 중요한 질문을 낳습니다.

> 4.03%가 의미적 interaction의 실패를 의미하는가, 아니면 scene-level similarity가 너무 강해서 pairwise ranking 자체가 망가지는 현상을 의미하는가?

이 문제를 해결하려면 **within-pair normalized metric** 또는 scene-shared component를 제거한 분석이 있으면 훨씬 강해집니다.

---

# 7. “7.2배 개선”은 현재 논문의 가장 위험한 표현

논문 후반부에서 rank-32 bilinear transformation

$$
S_W=v^T Wt
$$

을 사용하면 29.19%, 즉 cosine의 7.2배라고 합니다. 또한 rank-1 random vector는 0.44%, label-shuffled probe는 0.69%, full \(W\)는 23.91%라고 보고합니다. 

이것은 매우 흥미로운 control입니다.

하지만 저는 **이 결과를 논문의 핵심 evidence로 사용하면 오히려 논문이 위험해진다고 봅니다.**

이유는 간단합니다.

29.19%는 여전히 높은 성능이 아닙니다.

Random chance = 16.67%이므로 개선은 약 12.5 percentage points입니다.

즉,

> cosine interface가 완전히 잘못되었다

는 것까지 보여주지는 않습니다.

그리고 “7.2×”는 절대적인 성능 차이를 과장해서 보이게 합니다.

4.03 → 29.19는 7.2배지만,

**+25.16 percentage points**

라고 쓰는 것이 과학적으로 더 정직합니다.

더구나 full-rank가 오히려 낮다는 결과는 흥미롭지만, 왜 그런지 설명이 없습니다. 262,144 parameters와 rank-32의 parameter count 차이, regularization, optimization, sample size, GroupKFold 설정 등을 더 명확히 해야 합니다.

---

# 8. rank-32 실험의 가장 큰 문제: “새로운 scorer”와 “증거”가 혼합되어 있다

논문은 상당히 좋은 방어를 넣었습니다.

* GroupKFold
* unseen concepts
* random transformation
* label-shuffled probe
* full-rank transformation

등을 사용합니다. 

이 때문에 단순 overfitting이라고 공격하기는 어렵습니다.

하지만 리뷰어 입장에서는 여전히:

> “결국 CLIP embedding을 가지고 새로운 bilinear scorer를 학습해서 성능을 높였다는 것 아닌가?”

라고 생각할 수 있습니다.

그리고 이것은 사실입니다.

따라서 이 실험의 정확한 결론은

> “CLIP embedding 자체에 cross-modal information이 없지는 않으며, cosine이라는 고정된 scoring function이 이를 충분히 활용하지 못한다.”

정도입니다.

반대로

> “병목은 representation이 아니다.”

라고 단정하는 것은 과합니다.

왜냐하면 새로운 scorer가 정보를 잘 활용하려면 representation 자체에도 충분한 정보가 있어야 하지만, 그 사실만으로 representation이 충분하다고 증명되는 것은 아니기 때문입니다.

특히 image probe 자체가 0.759밖에 안 됩니다.

따라서 논문의 결론은

**“representation is sufficient”가 아니라 “representation contains usable information that is not fully exploited by cosine”**

정도가 훨씬 방어력이 높습니다.

---

# 9. 기존 논문들과의 novelty는 “있지만, 매우 좁다”

이 부분은 중요합니다.

현재 참고문헌들을 보면 이미 다음과 같은 방향들이 존재합니다.

* NegBench: negation failure 자체의 체계적 benchmark 
* Quantmeyer et al.: CLIP 내부에서 negation processing을 probing 
* Sammani et al.: embedding space의 negation direction과 steering 
* Aggarwal et al.: text embedding arithmetic을 통한 negation correction 
* SpaceVLM: negation을 point가 아닌 subspace로 모델링 
* PeakPatch: intermediate representation에 존재하는 negation signal을 최종 embedding에서 복구 
* DCSM 계열: CLIP cosine 자체가 compositional semantics를 충분히 반영하지 못한다는 geometry 관점 

그러므로

> “CLIP은 negation 정보를 가지고 있지만 cosine이 활용하지 못한다”

는 명제만으로는 novelty가 부족합니다.

논문의 novelty는 훨씬 좁게 잡아야 합니다.

제가 인정할 수 있는 novelty는:

> **기존의 “representation vs. similarity” 논쟁을 동일한 2×2 controlled counterfactual setting에서 \(\alpha,\beta,\gamma\)로 분해하고, negation success를 \(\gamma>\max(|\alpha|,|\beta|)\)라는 명시적 조건으로 연결했다.**

입니다.

이것은 **새로운 architecture도 아니고 새로운 benchmark도 아니며 새로운 learning algorithm도 아닙니다.**

대신 **diagnostic framework + empirical finding**입니다.

학부논문경진대회에서는 이것도 충분히 가능하지만, 그에 맞는 수준으로 claim을 낮춰야 합니다.

---

# 10. Winoground와의 관계는 오히려 논문을 강화할 수 있다

Winoground를 단순 참고문헌으로 넣는 것보다 적극적으로 활용하는 편이 좋습니다.

Winoground의 핵심은 identical word set을 유지하고 image-caption pairing을 평가한다는 것입니다. 

그리고 group score의 chance가 16.67%입니다. 

현재 논문의 2×2 matching도 본질적으로 동일한 구조를 갖습니다.

따라서 논문의 가장 좋은 framing은 다음과 같습니다.

> Winoground asks whether a model can correctly combine two independently meaningful states. We specialize this diagnostic structure to object presence negation and decompose the resulting four-way scores into modality main effects and a cross-modal interaction.

그러면 논문이 갑자기 훨씬 자연스러워집니다.

즉,

**Winoground → compositional matching failure**

↓

**본 연구 → negation-specific compositional matching**

↓

**2×2 factorial decomposition → 왜 failure가 발생하는지 정량화**

라는 흐름입니다.

이렇게 하면 단순히 “또 하나의 negation benchmark”라는 인상을 피할 수 있습니다.

---

# 11. 통계적으로도 몇 가지 불편한 부분이 있다

첫째, 논문은 2,480쌍을 사용하지만 개념 단위 평균을 강조합니다. 이것은 좋은 선택입니다. 그러나 독립 표본의 단위가 무엇인지 더욱 명확해야 합니다.

같은 concept, 같은 scene, 같은 template에서 여러 샘플이 나오기 때문에 **2,480개를 독립적인 observation처럼 취급하면 pseudo-replication 문제가 생깁니다.**

42 concepts가 실질적인 higher-level unit이라면 confidence interval과 significance testing 역시 concept-level 또는 hierarchical bootstrap을 기본으로 해야 합니다.

논문에는 42개 concept에 대한 bootstrap이 있지만, 이것이 본문의 모든 통계량에 일관되게 적용되는지 명확하지 않습니다.

둘째, 378개의 concept-model combinations는 사실상 독립적인 378개의 데이터 포인트가 아닙니다.

따라서

> “368/378”

이라는 숫자는 descriptive statistic으로는 좋지만 inferential evidence로 사용해서는 안 됩니다.

---

# 12. “9개 모델에서 모두 실패”는 생각보다 강한 결과가 아니다

논문은 3 architecture × 3 dataset × 2 objective × 4 fine-tuning이라고 설명하면서 9개 모델에서 같은 현상이 나타난다고 합니다. 

하지만 여기서 모델 다양성의 효과를 과장하면 안 됩니다.

실제로 같은 CLIP 계열을 기반으로 한 모델이라면 이들은 독립적인 evidence가 아닙니다.

오히려 좋은 것은:

> “The same ordering was observed across nine model configurations.”

정도입니다.

“일반적인 VLM 특성이다”라고 확대하면 안 됩니다.

특히 현재 실험이 사실상 CLIP/OpenCLIP 계열에 집중되어 있다면 제목부터 “Vision-Language Models”라고 넓히는 것보다 **“CLIP-based dual encoders”**라고 한정하는 편이 정확합니다.

---

# 13. 가장 큰 논리적 과장: “병목은 표현도 아니고 필요한 구조의 복잡도도 아니다”

이 문장은 삭제하거나 약화하는 것을 권합니다.

현재 실험이 보여주는 것은:

1. 일부 정보가 linear probe로 검출됨.
2. cosine에서는 interaction이 작음.
3. bilinear scorer로 일부 개선 가능.

여기까지입니다.

그런데 이것을

> “병목은 representation이 아니다.”

라고 바꾸면 논리적으로 한 단계 더 나갑니다.

왜냐하면 **probeability ≠ sufficiency**이기 때문입니다.

Linear probe가 0.759라는 것은 “정보가 있다”를 어느 정도 보여주지만, representation이 downstream negation matching에 필요한 충분한 정보를 가지고 있다는 것은 아닙니다.

오히려 현재 논문 자체의 결과가 이를 보여줍니다.

* text: 상당히 잘 probe됨
* image: 상대적으로 약함
* cosine: 거의 실패
* bilinear: 일부 회복

따라서 더 정확한 표현은:

> **“The failure cannot be explained by the complete absence of negation-related information in the representations; rather, the information available to the frozen embeddings is insufficiently exploited by cosine similarity.”**

정도입니다.

이 표현이면 기존 연구들과도 충돌이 적습니다.

---

# 14. 현재 논문에서 제가 가장 높게 평가하는 부분

의외로 **수식 자체가 아니라 experimental question의 분리**입니다.

논문은 사실 다음 세 질문을 구분합니다.

① 이미지에 객체 존재 정보가 있는가?

② 텍스트에 어느 객체가 부정되었는지에 대한 정보가 있는가?

③ 그 두 정보가 cross-modal similarity에서 올바르게 결합되는가?

기존 연구는 이 세 가지를 종종 섞습니다.

현재 논문은 이것을 분리하려고 합니다.

특히

> “선형적으로 검출되는 것과 원래 similarity ranking을 결정할 수 있는 것은 다르다.”

라는 문제제기는 상당히 좋습니다. 

이 한 문장이 사실 논문의 가장 좋은 intellectual contribution입니다.

---

# 15. 반대로 제가 Reject할 때 쓰는 결정적 사유

리뷰어 리뷰 양식으로 작성한다면 대략 이렇게 쓰겠습니다.

**Strengths**

* Negation understanding을 단순 benchmark 문제보다 cross-modal interaction 문제로 재정의한 점이 흥미롭다.
* 2×2 factorial decomposition은 문제를 명확하게 정식화한다.
* 동일 장면 counterfactual pair를 사용하여 scene-level confound를 줄이려는 설계가 적절하다.
* linear probe와 cosine matching을 분리하여 representation과 scoring interface를 구별하려는 분석이 유용하다.
* Winoground와 유사한 controlled matching 구조를 negation-specific setting으로 확장한 점이 흥미롭다.

**Weaknesses**

1. \(\gamma\)를 semantic cross-modal interaction으로 해석하는 근거가 충분하지 않다.
2. BEAF inpainting artifact가 image-side probe와 interaction coefficient에 영향을 줄 가능성이 크다.
3. AB-swap이 token multiset을 통제하더라도 syntax/template/lexical shortcut을 완전히 제거하지 못한다.
4. probe performance가 downstream sufficiency를 의미하지 않는데, 논문의 결론은 이를 다소 과하게 일반화한다.
5. bilinear scoring의 29.19%는 cosine보다 크게 높지만 절대적으로는 여전히 낮아, “cosine이 병목”이라는 강한 결론을 충분히 뒷받침하지 못한다.
6. Winoground 및 최근 negation representation 연구와의 차별성이 충분히 명확하게 정리되지 않았다.
7. 2페이지 안에 너무 많은 분석을 넣으면서 핵심 주장과 보조 분석의 우선순위가 흐려진다.

---

# 16. 최종 점수

제가 학부논문경진대회에서 실제로 평가한다면 다음과 같이 주겠습니다.

| 항목               |         점수 |
| ---------------- | ---------: |
| 문제의 중요성          |       8/10 |
| 아이디어             |       8/10 |
| 방법론적 명확성         |       7/10 |
| 실험 설계            |       6/10 |
| 실험적 검증           |       6/10 |
| 기존 연구 대비 novelty |       6/10 |
| 논리적 엄밀성          |       5/10 |
| 논문 구성            |       6/10 |
| 재현성/설명 충분성       |       5/10 |
| 종합               | **6.3/10** |

**판정: Weak Reject / Borderline**

학부논문경진대회라면 일반 학회보다 기준을 약간 낮춰 **Borderline Accept**까지 고려할 수 있지만, 사용자가 제시한 것처럼 “양질의 논문만을 억셉하고 acceptance rate를 엄격하게 관리한다”는 리뷰 정책이라면 저는 **현재 버전은 Reject**하겠습니다.

다만 **아이디어 자체를 Reject하는 것은 아닙니다.**

오히려 한두 가지 실험만 제대로 보강하면 **Accept 쪽으로 이동할 가능성이 상당히 높은 논문**입니다.

가장 중요한 것은 세 가지입니다.

**첫째, BEAF artifact 검증.**
이것이 없으면 image-side evidence가 약합니다.

**둘째, \(\gamma\)가 정말 object-specific interaction인지 검증.**
현재의 permutation control을 더 명확하게 제시해야 합니다.

**셋째, claim을 낮추기.**
“representation은 문제가 아니다”가 아니라 **“representation에 정보가 전혀 없는 것은 아니며, cosine interface가 그 정보를 충분히 활용하지 못한다”**로 바꾸는 것이 훨씬 강합니다.

그리고 2페이지라는 제약을 고려하면, 저는 **rank-32 7.2× 실험과 T2I 외부과제보다 2×2 decomposition + controlled probe + artifact control에 지면을 집중**시키겠습니다. 지금 논문의 가장 강한 논리는 “새로운 방법으로 성능을 높였다”가 아니라 **“왜 linear probe가 성공하는데 cosine matching은 실패하는지를 동일한 controlled experiment에서 분리해서 보여준다”**는 데 있기 때문입니다.

현재 상태의 원고는 “분석 결과가 많은 논문”에 가깝지만, 위 세 부분을 보완하면 **“하나의 반증 가능한 주장을 여러 독립적인 실험으로 검증한 진단 논문”**으로 바뀔 수 있습니다. 그 차이가 이 논문을 Reject와 Accept 사이에서 가르는 핵심입니다.
