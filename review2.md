전체적으로 보면, 이 논문은 “틀린 논문”이라기보다 핵심 아이디어가 흥미로운 반면, 현재 서술 수준에서는 주장보다 증거가 한 단계 약합니다. 특히 강한 부분은 통제된 최소쌍과 2×2 분해이고, 가장 치명적인 약점은 그 분해를 “메커니즘 설명”으로 해석하는 데 있습니다.

제 판단으로는 현재 상태에서 메이저 리비전입니다. 특히 CVPR/ICML급 기준으로는 “좋은 진단 논문”이 될 가능성은 있지만, “새로운 메커니즘을 밝혀냈다”는 수준의 주장을 유지하기에는 아직 부족합니다.

### 1. 핵심 기여에 대한 평가

논문의 가장 좋은 부분은 문제를 “CLIP에 negation 정보가 있느냐 없느냐”가 아니라 “네 개의 매칭 점수에서 주효과와 상호작용 중 무엇이 순위를 지배하느냐”로 바꾼 것입니다. 논문은 이미지 상태 \(a\in\{+1,-1\}\), 텍스트 상태 \(b\in\{+1,-1\}\)를 두고

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

로 2×2 점수를 완전히 분해합니다. 네 개의 실수에 대한 포화된 2×2 요인분해라는 점에서 수학적으로 깔끔하고, 실제 성공 조건

$$
\min(S_{++},S_{--})>\max(S_{+-},S_{-+})
\iff
\gamma>\max(|\alpha|,|\beta|)
$$

도 맞습니다. 논문이 주장하듯 이는 Winoground의 group score를 같은 좌표계로 다시 표현한 것입니다. 논문 스스로도 이 규칙 자체가 새롭지 않음을 인정하고 있습니다. 

문제는 바로 여기서 시작됩니다.

이 식은 “새로운 설명 모델”이라기보다 “네 개 점수의 재매개변수화”입니다. 논문도 “가정 없는 좌표 변환”이라고 표현합니다.  그러므로 \(\gamma\)를 곧바로 “결합 능력(binding)” 또는 \(\alpha,\beta\)를 “BoW 성분”이라고 부르는 것은 수학적으로 자동으로 따라오는 결론이 아닙니다.

즉,

> \(\gamma \neq 0\)

은 “두 변수가 상호작용하는 점수항이 존재한다”는 뜻이지,

> “모델이 극성과 객체의 의미적 결합을 이해한다”

는 뜻은 아닙니다.

이 구분이 이 논문의 가장 중요한 약점입니다.

---

### 2. 가장 큰 문제: algebraic interaction ≠ semantic understanding

논문은 \(\gamma\)를 “극성-객체 결합의 정량적 정의”로 둡니다.  그런데 이것은 정의를 그렇게 붙인 것이지, \(\gamma\)의 원인이 실제 semantic binding이라는 증거는 아닙니다.

예를 들어 \(\gamma\)가 생기는 이유는 여러 가지일 수 있습니다.

텍스트 길이와 위치 효과, 특정 토큰의 위치, 이미지 편집 잔여물, 객체 종류별 데이터 분포, CLIP의 비선형 saturation, 부정 표현이 특정 문법 패턴과 결합되는 현상 등도 2×2 interaction으로 나타날 수 있습니다.

결국 이 논문의 데이터가 직접 증명하는 것은

> “부정 정답 여부가 주효과보다 충분히 강한 interaction을 가져야 한다.”

이지,

> “그 interaction이 semantic negation binding이다.”

까지는 아닙니다.

이 점은 [CLIP Behaves like a Bag-of-Words]와 비교하면 더 선명합니다. 그 논문은 unimodal embedding 자체에서 attribute-object binding이 선형적으로 추출 가능하고, cross-modal misalignment가 문제라고 주장합니다. 실제로 LABCLIP이 text embedding을 선형 변환했을 때 probe 방향의 cosine similarity가 크게 증가합니다. 

반면 현재 논문의 \(\gamma\)는 “상호작용이 존재하는가”를 보여주지 “그 상호작용의 의미적 정체가 무엇인가”를 보여주지 않습니다.

따라서 논문의 표현은

> “\(\gamma\) is an interaction term associated with polarity–object matching”

정도가 안전하고,

> “\(\gamma\) is the quantitative definition of binding”

은 과합니다.

---

### 3. 프로빙 결과도 “정보가 존재한다”를 완전히 입증하지 못한다

논문은 객체별 probe와 존재 탐지 AUC를 사용해 기존의 “정보는 이미 표현돼 있다”는 가설을 통제된 자연 이미지에서도 확인했다고 주장합니다. 이미지 probe 0.630, 텍스트 probe 0.912, 존재 탐지 macro AUC 0.759를 제시합니다. 

하지만 여기에도 상당한 논리적 비약이 있습니다.

특히 텍스트 쪽은 AB-swap을 사용합니다. 두 문장은 거의 같은 token multiset을 공유하지만 결합만 바꿉니다.  이것은 기존의 “부정 표지 존재 여부” shortcut을 제거한다는 점에서는 좋습니다.

하지만 probe가 높은 성능을 내는 것이 “의미론적 polarity-object binding” 때문인지, 특정 위치·구문 패턴 때문인지는 별도의 문제입니다.

예컨대 다음 두 문장의 차이를 생각할 수 있습니다.

> truck positive, train negative
> train positive, truck negative

이 구분 자체가 매우 강한 positional/syntactic signal을 가지고 있습니다. 따라서 probe가 “어떤 객체가 부정되었는가”를 잘 맞힌다는 사실은 표현공간에 관련 정보가 남아 있다는 증거이기는 하지만, 그것이 semantic representation인지 surface structural encoding인지는 분리되지 않았습니다.

이 점은 논문이 비판하는 [2]의 probing shortcut과 사실상 같은 종류의 위협을 완전히 제거했다고 보기는 어렵습니다.

---

### 4. 이미지 최소쌍 설계는 좋지만, 인페인팅 confound가 아직 남는다

BEAF의 동일 장면에서 객체 하나를 제거한 최소쌍을 사용하는 것은 이 논문의 가장 설득력 있는 실험 설계입니다. 33개 개념, 1,357쌍이라는 규모도 작은 편은 아닙니다. 

또한 저자들은 placebo test를 수행했고, 오염된 것으로 판단한 개념을 제거하면 오히려 \(\gamma\)와 AUC가 올라간다고 보고합니다. 

그러나 논문 스스로 인정하듯 이것으로 inpainting artifact 문제가 완전히 해결되지는 않습니다. 실제 객체가 원래부터 없는 자연 이미지와 비교하지 않았습니다. 

여기서 중요한 점은 “artifact가 있느냐”와 “artifact가 \(\gamma\)를 만들고 있느냐”가 다른 문제라는 것입니다.

예를 들어 트럭 제거로 생긴 특정 영역의 texture 변화가 “truck-related image state”를 encode하고 있다면, placebo 객체 Y가 영향을 받지 않는다고 해서 truck-specific artifact를 배제할 수 없습니다. 지금의 placebo test는 “모든 artifact가 무관하다”를 보여주는 것이 아니라 “다른 객체의 신호가 전부 같은 방식으로 오염돼 있지는 않다” 정도입니다.

따라서 stronger control이 필요합니다.

가장 좋은 추가 실험은 “BEAF edited image vs 원래부터 객체가 없는 real image”를 matching하여, 객체 제거라는 intervention 자체와 실제 absence를 분리하는 것입니다.

---

### 5. [Similarity Is Not Logic]과의 차별성이 생각보다 약하다

이 부분은 리뷰어가 가장 집요하게 물을 가능성이 높습니다.

[Similarity Is Not Logic]은 이미 dual-encoder의 scalar similarity가 Boolean operator를 제대로 실행하지 못하고, concept evidence를 soft pooling하는 현상을 진단합니다. 또한 operator-dependent signal은 존재하지만 ranking에 충분하지 않다고 보고하고, LCSE로 evidence extraction과 constraint execution을 분리합니다.  

현재 논문은 이와의 관계를 상당히 솔직하게 인정합니다. 실제로 “새로운 결론이 아니라 기존 결론의 좌표계”라고 스스로 표현합니다. 

그렇다면 새로움은 결국 다음 두 가지입니다.

첫째, 자연 이미지의 객체 제거 최소쌍으로 통제를 강화했다는 점.

둘째, 기존의 qualitative diagnosis를 \(\gamma/\max(|\alpha|,|\beta|)\)라는 continuous quantity로 바꾸었다는 점.

둘 다 의미가 있지만, “mechanistic breakthrough”라고 부르기에는 부족합니다.

더 심각한 문제는 LCSE와 직접 비교하지 않았다는 점입니다.

[Similarity Is Not Logic]은 FACTOR-Bench에서 LCSE가 85.5%, SigLIP 2에서 90.7%, NegBench MCQ에서 27.2%→65.2% 개선을 보고하며, 핵심적으로 “similarity 자체로 logic을 수행하려 하지 말고 constraint execution을 별도로 하자”는 입장입니다. 

그런데 현재 논문에서는 정작 자신의 BEAF/AB-swap setting에서 LCSE가 얼마나 되는지 보여주지 않습니다.

이건 큰 결손입니다.

논문의 핵심 질문이 “왜 2×2 matching이 무너지는가”라면 LCSE는 거의 필수 baseline입니다. 최소한 다음을 같은 benchmark에서 비교해야 합니다.

CLIP
NegCLIP / NegFull
LABCLIP
LCSE
단순 text/image steering
rank-1 bilinear
rank-2 bilinear

현재는 관련 논문을 잘 읽었지만, 중요한 경쟁 가설을 실제 동일 조건에서 비교하지 않았습니다.

---

### 6. [CLIP Behaves like a Bag-of-Words]와의 관계는 흥미롭지만, 오히려 논문의 framing을 약화시킨다

[CLIP Behaves like a Bag-of-Words]는 “CLIP은 cross-modally BoW처럼 보이지만 unimodally는 BoW가 아니다”라는 결론을 내리고, LABCLIP으로 cross-modal alignment를 개선합니다. 

현재 논문은 이를 \(\alpha,\beta,\gamma\)의 언어로 재해석합니다.

* \(\alpha,\beta\): dominant main effects
* \(\gamma\): cross-modal interaction

이 해석 자체는 상당히 좋습니다. 실제로 두 논문의 결과를 모순이 아니라 같은 현상의 서로 다른 항으로 정리한 것은 이 논문의 장점입니다. 논문도 이를 명시합니다. 

그러나 여기서도 중요한 반례가 있습니다.

LABCLIP은 attribute-object binding에는 효과가 있었지만, 같은 종류의 linear transformation을 negation에 적용하자 in-sample 82.64%가 holdout 11.57%로 떨어졌습니다. 

이 결과는 현재 논문의 핵심 주장을 지지하는 동시에, 더 근본적인 질문을 남깁니다.

왜 attribute-object binding에서는 cross-modal linear alignment가 작동하고 polarity-object binding에서는 작동하지 않는가?

현재 논문은 “rank-1 bilinear에서는 두 검출기의 부호가 동시에 뒤집혀야 하기 때문”이라고 설명합니다. 

이것은 좋은 가설이지만, 현재 증거는 이 설명을 검증했다기보다는 해당 구조와 일치하는 것입니다. 이 차이를 실험적으로 분리하지 않았습니다.

---

### 7. [Winoground]와의 연결은 정확하지만, novelty claim에는 제약이 있다

Winoground의 group score는 본질적으로 두 image-caption pairing을 동시에 맞추는 문제이고, 원 논문에서도 CLIP ViT-B/32가 text 30.75%, image 10.50%, group 8.00%로 random chance를 넘지 못했습니다. 

현재 논문이 하는 일은 Winoground의 세 점수를 \(\alpha,\beta,\gamma\)로 정확하게 표현하는 것입니다. 이건 유용한 이론적 정리입니다.

하지만 논문 스스로 인정하듯 이 규칙 자체는 새로운 평가규칙이 아닙니다. 

따라서 논문의 novelty는

> “Winoground를 개선했다”

가 아니라

> “Winoground-style 2×2 matching failure를 factorized score coordinates로 해석할 수 있다”

정도로 제한해야 합니다.

특히 “이것이 negation에 국한되지 않는다”는 주장은 현재 데이터가 받쳐주지 않습니다. 실제 평가된 것은 essentially one family of object-presence negation입니다. 논문도 한계에서 관계부정, 행위부정, single-object negation을 제외했다고 명시합니다. 

---

### 8. NegBench에 대한 가장 중요한 문제: 이 논문의 task는 NegBench보다 훨씬 좁다

[Vision-Language Models Do Not Understand Negation]의 NegBench는 retrieval과 MCQ를 포함하고, 이미지·비디오·의료 데이터까지 확장하며 79k examples를 사용합니다. 

특히 그 논문은 모델이 affirmation/negation/hybrid 형태에서 매우 심하게 실패한다는 것을 보였고, VOC2007에서 예를 들어 affirmation 82% 대비 negation 3%라는 극단적인 차이도 보고했습니다. 

현재 논문은 이러한 광범위한 negation problem을 다루지 않습니다. 오직

> 동일 장면 + 특정 객체 존재/부재 + 다른 객체와 결합된 negation

이라는 매우 특정한 subset입니다.

따라서 제목과 초록에서 “CLIP negation retrieval failure”를 일반적인 negation problem으로 읽히게 만들면 과장입니다.

더 정확한 표현은

> negated object retrieval under controlled multi-object minimal-pair conditions

에 가깝습니다.

저자도 이 한계를 인정하고 있으므로, 오히려 본문 초반에서 범위를 명확히 제한하는 편이 논문을 더 강하게 만들 것입니다. 

---

### 9. [When Negation Is a Geometry Problem]과의 관계에서 가장 중요한 것은 오히려 반증 결과다

CVPR 2026의 geometry paper는 CLIP embedding space에 “negation direction”이 존재할 수 있으며, representation steering으로 fine-tuning 없이 개선할 수 있다고 주장합니다.  

현재 논문은 방향 정렬 실험을 직접 해보고, 이론상 9.37배 \(\gamma\) 증가가 필요하지만 실제로는 약 4.85배에 그쳤다고 보고합니다. 또한 실제 이동 방향의 alignment가 1.0이 아니라 약 0.52였다고 분석합니다. 

이건 좋은 대조 실험입니다.

하지만 여기서도 causal claim을 조심해야 합니다.

현재 결과는

> “이 particular representation rotation strategy가 실패했다.”

를 보여주는 것이지,

> “geometry hypothesis가 틀렸다.”

를 보여주는 것은 아닙니다.

실제로 geometry paper도 steering strength를 잘못 크게 잡으면 semantic structure가 붕괴한다고 보고합니다. 

따라서 현재 논문이 “direction alignment를 배제했다”고 강하게 표현하는 것은 범위가 너무 넓습니다. 보다 정확하게는

> the tested probe-normal alignment intervention does not produce the predicted \(\gamma\) amplification

정도로 써야 합니다.

---

### 10. 가장 흥미로운 결과인 rank-1/rank-2 bilinear은 오히려 논문의 결론을 약화시킨다

표 5가 매우 중요합니다.

공유 \(W\)의 rank를 키우면:

* identity: 0.88%
* diagonal: 2.58%
* rank-1: 22.11%
* rank-2: 22.99%
* rank-32: 15.25%
* full: 9.65%

가 됩니다. 

이 결과는 “단순 cosine interface가 문제다”라는 주장에는 매우 강한 힌트입니다.

그런데 통계적 결론은 불완전합니다. 최선 rank-2의 22.99%도 저자들이 인정하듯 macro 기준에서 one-sided \(p\approx0.057\)입니다. 즉 “chance보다 낫다”고 명시적으로 결론내리기엔 경계선입니다. 

더구나 rank-32와 full이 다시 악화됩니다. 따라서 “cross-dimensional structure가 필요하다”는 해석보다는

> 특정 저차원 구조가 이 작은 데이터에서는 일반화에 유리하다.

정도가 현재 증거에 맞습니다.

특히 262,144개의 full \(W\)가 overfit하는 것은 아주 예상 가능한 결과입니다. 이것을 “interface가 더 강해질수록 오히려 나빠진다”는 메커니즘처럼 읽으면 위험합니다. 샘플 수가 1,357개이고 parameter가 262k이기 때문입니다.

---

### 11. “무작위 이하 성능은 결정론적 역전” 주장은 맞지만, 중요도가 과대평가되어 있다

논문은 \(\max(|\alpha|,|\beta|)>\gamma\)일 때 dominant main effect가 score ordering을 결정해서 accuracy가 chance 이하로 내려간다고 설명합니다. 

이건 해당 2×2 scoring setup에서는 맞습니다.

하지만 이것이 기존 “below chance” 결과를 설명하는 것 이상의 의미를 갖는지는 별개입니다.

왜냐하면 2×2 score에서 네 값의 ordering을 정의하고 나면, 이런 inversion 현상은 상당 부분 구조적으로 따라오기 때문입니다. 즉 “기존 논문들이 왜 below chance인지 설명했다”는 측면은 좋지만, 이것이 CLIP의 내부 학습과정이나 inference mechanism에 대한 독립적인 causal explanation은 아닙니다.

---

### 12. 실험 통계는 대체로 조심스럽지만, 모델 수 9개를 가지고 하는 일부 분석은 약하다

저자들은 이 문제를 상당히 잘 인식하고 있습니다.

특히 \(\gamma/\max(|\alpha|,|\beta|)\)와 2×2 accuracy의 Pearson/Spearman correlation이 각각 0.845/0.883이지만 이것은 같은 네 개 점수로부터 만들어진 양이라 사실상 구조적으로 유도된다고 스스로 인정합니다. 

이건 좋은 자기비판입니다.

반대로 존재 탐지 AUC와 2×2 accuracy의 correlation이 \(r=0.010\)이라는 결과는 더 가치 있습니다. 서로 다른 공간의 서로 다른 지표이기 때문입니다. 

다만 \(n=9\) 모델로 “predictive validity”를 논하는 것은 원래 매우 약합니다. 동일한 backbone family와 fine-tuning lineage가 중복되고, 모델들이 독립 표본이 아니기 때문입니다. 논문은 이것도 인정합니다. 

따라서 이 figure는 supplementary에 가까운 역할이어야 합니다.

---

### 13. paper 자체의 형식적 완성도에도 문제가 있다

이건 연구 내용과 별개로 실제 리뷰에서 지적할 만합니다.

첫 페이지에 제목이 `PAPER_2`로 되어 있고 저자/소속이 “미정”입니다. 또 본문에

> `[Figure 1 — 직접 그려주십시오]`

라는 작업 메모가 그대로 남아 있습니다.  

표 4도 현재 문서에서는 실제 표 내용보다 설명 문장이 이어지는 형태로 보입니다. 이 상태는 workshop draft라면 이해되지만, 정식 conference submission이라면 명백한 presentation issue입니다.

더 중요한 것은 제목/섹션 numbering도 일부 어색합니다. “7. 한계 → 8. 결론”의 위치와 실제 본문 구조가 자연스럽지 않습니다. 

내용이 좋은데 이런 형식적 흔적 때문에 실험 논문의 신뢰성이 깎일 수 있습니다.

---

## 제가 리뷰어라면 핵심 Weakness를 이렇게 적겠습니다

**W1. The proposed factorization is algebraically exact but not mechanistically identifying.**
The decomposition of four scores into \(C,\alpha,\beta,\gamma\) is a saturated 2×2 reparameterization. While \(\gamma\) measures score interaction, the paper has not established that it corresponds specifically to semantic polarity–object binding rather than syntactic, positional, dataset, or image-editing artifacts.

**W2. The probing experiments do not fully establish semantic information preservation.**
The AB-swap text setup removes a trivial negation-token shortcut, but a probe can still exploit positional or syntactic cues. Likewise, BEAF-based image probing can contain edit-specific residual signals.

**W3. The comparison to the most relevant prior methods is incomplete.**
Most importantly, LCSE from *Similarity Is Not Logic* is not evaluated on the proposed controlled benchmark. Since LCSE explicitly separates atomic evidence extraction from Boolean execution, it is a direct competing hypothesis to the paper’s interpretation of the bottleneck.

**W4. The empirical scope is substantially narrower than the framing suggests.**
The benchmark covers object-presence negation in multi-object scenes, not relation/action negation, single-object absence, or broad natural-language negation as evaluated in NegBench.

**W5. The evidence for a general bilinear-interface remedy is inconclusive.**
Rank-1/rank-2 shared \(W\) improves point estimates, but the best macro result is not statistically separated from chance under the paper’s own testing protocol.

**W6. Some intervention conclusions are stronger than the experiments justify.**
Failure of the tested steering or scaling interventions should not be interpreted as evidence that the corresponding broader class of representation-space interventions is impossible.

---

## Major Strengths

반대로 reviewer가 인정할 부분도 분명합니다.

첫째, 문제 설정이 명확합니다.
둘째, 동일 장면 최소쌍을 사용한 통제가 기존의 “정보가 있다”류 probing보다 훨씬 낫습니다.
셋째, \(\gamma/\max(|\alpha|,|\beta|)\)는 적어도 해당 benchmark 내부에서 failure severity를 표현하는 유용한 diagnostic quantity입니다.
넷째, 기존 문헌 간의 겉보기 모순을 하나의 좌표계에서 연결하려는 시도는 상당히 좋습니다. [Similarity Is Not Logic]의 main-effect dominance와 [CLIP Behaves like a Bag-of-Words]의 unimodal binding-presence를 각각 \(\alpha,\beta\)와 \(\gamma\)로 연결한 부분은 이 논문의 가장 설득력 있는 conceptual contribution입니다. 

---

## 가장 중요한 수정 실험 5개

이 논문을 실제로 살리려면 저는 다음을 우선순위로 봅니다.

1. **동일 benchmark에서 LCSE를 직접 실행**
   이것이 가장 중요합니다. 현재 논문의 핵심 diagnosis가 기존 LCSE보다 무엇을 더 설명하는지 보여줘야 합니다.

2. **실제 absence image control 추가**
   BEAF 제거 이미지와 원래부터 해당 객체가 없는 이미지의 artifact/control 실험이 필요합니다.

3. **text-side shortcut probing 추가**
   AB-swap에서 객체 identity, token position, syntax-only feature를 제거하거나 통제한 probe를 추가해야 합니다.

4. **외부 benchmark correlation 검증**
   \(\gamma/\max(|\alpha|,|\beta|)\)가 실제 NegBench나 다른 compositional benchmark 성적을 예측하는지 검증해야 합니다. 논문 자체도 이것을 아직 하지 않았다고 인정합니다. 

5. **rank-1/rank-2 bilinear에 대한 cluster-level significance와 더 많은 concepts**
   현재 33 concepts는 특히 low-dimensional intervention의 일반화를 판단하기엔 부족합니다.

---

## 최종 판정

현재 버전이라면 저는 **Weak Reject / Borderline Major Revision** 쪽입니다.

점수로 표현하면 대략:

**Technical quality: 6/10**
**Novelty: 6/10**
**Empirical evidence: 5/10**
**Clarity: 6/10**
**Significance: 7/10**

입니다.

다만 중요한 점은, 이 점수는 “아이디어가 약해서” 낮은 것이 아닙니다. 오히려 아이디어는 꽤 좋습니다. 문제는 논문이 보여준 것보다 한 단계 더 큰 주장을 하고 있다는 것입니다.

가장 안전하고 강한 버전의 논문은 다음 주장입니다.

> **“Controlled natural-image minimal pairs reveal that CLIP’s negation retrieval failure is governed by a small cross-modal interaction term that is consistently dominated by image/text main effects. This factorization reconciles prior apparently conflicting findings about the presence versus usability of negation information, and provides a quantitative diagnostic for the failure of scalar similarity.”**

반대로 현재의

> “이것이 CLIP negation failure의 underlying mechanism이다”

라는 framing은 아직 과합니다.

그리고 **가장 큰 리뷰어 질문은 단 하나**일 가능성이 높습니다.

> “당신들의 \(\gamma\)는 정말 semantic binding의 증거인가, 아니면 단순한 2×2 score interaction을 새로운 이름으로 부른 것인가?”

이 질문에 실험적으로 답하지 못하면, 이 논문의 핵심 novelty가 상당 부분 “유용한 재표현” 수준으로 내려갑니다. 반대로 이 질문을 LCSE + 자연 absence control + shortcut-controlled probing으로 해결하면 논문의 수준이 상당히 올라갑니다. 
네. 다만 명칭을 그대로 쓰면 약간 위험합니다. 현재 식에서 \(\alpha-\gamma\), \(\beta-\gamma\)는 “전체적인 affirmation/presence preference”가 아니라, 각각 특정 조건에서 interaction으로 설명되지 않고 남는 주효과입니다.

### 1. \(\alpha-\gamma\), \(\beta-\gamma\)를 어떻게 정의할 것인가

현재 논문의 정의가

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

이고, \(a=+1\)은 객체 존재, \(a=-1\)은 객체 부재, \(b=+1\)은 긍정 caption, \(b=-1\)은 부정 caption입니다. 

그러면

$$
S_{-+}-S_{--}
=2(\alpha-\gamma).
$$

즉 **객체가 없는 이미지에서 positive caption과 negative caption의 차이**입니다.

따라서

$$
A_{\rm resid}\equiv \alpha-\gamma
$$

를

> residual affirmation preference

또는

> conditional affirmation preference

라고 부르는 것은 타당합니다.

마찬가지로

$$
S_{+-}-S_{--}
=2(\beta-\gamma),
$$

즉 **negative caption 조건에서 객체가 있는 이미지와 없는 이미지의 차이**입니다. 따라서

$$
P_{\rm resid}\equiv \beta-\gamma
$$

를

> residual presence preference

라고 부를 수 있습니다. 논문도 이미 이 두 항을 각각 “object가 없는 image에 남는 positive-caption preference”와 “negative caption에 대해 남는 object-presence preference”로 해석하고 있습니다. 

오히려 저는 **affirmation preference / presence preference라고 단독 명명하는 것보다는 `residual` 또는 `conditional`을 붙이는 것을 권합니다.**

왜냐하면 \(\alpha\) 자체는 unconditional main effect이고, \(\alpha-\gamma\)는 interaction을 제거하고 난 뒤의 특정 상태에서 남는 preference이기 때문입니다.

가장 깔끔한 정의는:

$$
\boxed{
A_{\rm resid}=\alpha-\gamma
=\frac{S_{-+}-S_{--}}{2}
}
$$

$$
\boxed{
P_{\rm resid}=\beta-\gamma
=\frac{S_{+-}-S_{--}}{2}
}
$$

입니다.

그리고 모델 간 비교를 한다면 논문이 이미 사용한 것처럼 scale-normalized quantity를 병기하는 편이 좋습니다. 논문도 \(r_{\rm text}=(\alpha-\gamma)/(\alpha+\gamma)\) 같은 정규화를 사용합니다. 

---

## 2. 그런데 더 중요한 것은 \(w_i,w_t\)와 \(d_i,d_t\)의 관계입니다

여기서는 네 개를 구분해야 합니다.

\(w_i,w_t\): **linear probe가 학습한 discriminative normal**

\(d_i,d_t\): **두 대립 상태의 평균 차이(direction)**

이 둘은 관련 있지만 일반적으로 같은 벡터가 아닙니다.

예를 들어

$$
d_i
=
\mu_{i,+}-\mu_{i,-},
\qquad
d_t
=
\mu_{t,+}-\mu_{t,-}
$$

라고 정의하면, \(d_i,d_t\)는 “positive와 negative 상태가 평균적으로 representation space에서 어느 방향으로 이동하는가”를 나타냅니다.

반면 \(w_i,w_t\)는 binary classification boundary의 법선입니다.

일반적인 Gaussian LDA 관점에서는

$$
w \propto \Sigma^{-1}d.
$$

따라서 covariance가 isotropic하거나 whitened representation이라면

$$
w\parallel d,
$$

하지만 일반적으로는

$$
w\not\parallel d.
$$

즉

$$
\cos(w,d)\approx1
$$

을 기대하려면 representation covariance까지 통제해야 합니다.

이 구분이 현재 논문에 상당히 중요합니다.

---

## 3. 이것이 현재 PAPER_4의 가장 중요한 논리적 연결점입니다

현재 논문은 이미 **probe normal을 alignment하는 것과 실제 embedding movement direction을 alignment하는 것은 다르다**고 관찰했습니다. probe 법선 정렬은 1.0까지 만들었지만 실제 이동 방향 정렬은 약 0.52에 그쳤고, 따라서 예상했던 \(\gamma\) 증폭의 절반 정도만 얻었습니다. 

이 결과는 사실상

$$
w_i \leftrightarrow d_i
$$

를 동일시하면 안 된다는 직접적인 empirical evidence입니다.

즉, 현재의 geometry argument에서는 **\(w_i,w_t\)보다 \(d_i,d_t\)가 더 직접적인 quantity**입니다.

---

## 4. 왜 \(\gamma\)가 \(d_i,d_t\)의 alignment와 연결되는가

논문이 제시한 local additive model은

$$
\gamma
\approx
\frac14 a_I a_T
\cos(d_I,d_T)
$$

형태입니다. 실제 논문에서도 이 형태로 \(\gamma\)를 설명하고 있습니다. 

이 식의 의미는 상당히 명확합니다.

$$
\boxed{
\gamma
\sim
\text{image polarity signal magnitude}
\times
\text{text polarity signal magnitude}
\times
\text{cross-modal directional alignment}
}
$$

즉 세 요소가 있습니다.

$$
a_I
$$

: image space에서 presence/absence를 구별하는 signal 크기

$$
a_T
$$

: text space에서 positive/negative를 구별하는 signal 크기

$$
\cos(d_I,d_T)
$$

: 그 두 signal이 cross-modal space에서 얼마나 같은 방향으로 정렬돼 있는지.

따라서 **\(d_i,d_t\)는 \(\gamma\)의 geometric interpretation과 직접 연결**됩니다.

반면 \(w_i,w_t\)는 “분류가 가능한 방향”이고, 반드시 cross-modal interaction 자체를 나타내지는 않습니다.

---

## 5. \(w\)와 \(d\)의 관계를 실험적으로 분리하면 논문이 훨씬 강해집니다

저라면 이걸 다음 세 단계로 실험합니다.

먼저 각 modality에서

$$
w_i,\quad w_t
$$

를 binary linear probe로 얻습니다.

동시에 평균 difference vector

$$
d_i=\mathbb E[z_i^+]-\mathbb E[z_i^-],
\qquad
d_t=\mathbb E[z_t^+]-\mathbb E[z_t^-]
$$

를 계산합니다.

그 다음 다음 세 cosine을 모두 보고합니다.

$$
\cos(w_i,d_i)
$$

$$
\cos(w_t,d_t)
$$

$$
\cos(d_i,d_t).
$$

여기서 각각의 의미가 다릅니다.

$$
\cos(w_i,d_i)
$$

는 “image binary classification direction과 actual embedding shift가 일치하는가?”

$$
\cos(w_t,d_t)
$$

는 text에서도 동일한 질문입니다.

그리고

$$
\cos(d_i,d_t)
$$

가 실제 cross-modal polarity alignment입니다.

이 세 개를 분리하면 현재 논문의 geometry story가 훨씬 명확해집니다.

---

## 6. 특히 중요한 반례: \(w_i\)와 \(w_t\)가 잘 정렬돼도 \(\gamma\)가 클 필요는 없다

이게 핵심입니다.

가령

$$
\cos(w_i,w_t)=1
$$

이라고 해도

$$
\cos(d_i,d_t)\ll1
$$

일 수 있습니다.

왜냐하면 \(w_i,w_t\)는 각각 자기 modality의 covariance structure를 반영하기 때문입니다.

예를 들어

$$
w_i=\Sigma_i^{-1}d_i,
\qquad
w_t=\Sigma_t^{-1}d_t
$$

라면 covariance가 서로 다르면 \(w_i,w_t\)의 alignment는 \(d_i,d_t\)의 alignment와 별개가 됩니다.

따라서 현재 논문에서

> “probe normals are aligned”

를

> “the polarity directions are aligned”

로 해석하면 논리적 비약입니다.

오히려 현재의 실패 결과는 이것을 잘 보여줍니다. 논문도 실제로 “회전은 probe normal을 정렬시키지만 \(\gamma\)가 요구하는 것은 embedding movement direction”이라고 명시합니다. 

---

## 7. [Similarity Is Not Logic]의 \(d_t\)와 연결하면 더 재미있습니다

[Similarity Is Not Logic]은 직접

$$
\Delta_X=g(\text{“not X”})-g(\text{“X”})
$$

를 정의하고, 그 magnitude와 concept 간 pairwise cosine을 측정했습니다. 논문에서는 magnitude 0.31, concept 간 cosine \(0.62\pm0.09\), 그리고 이 방향 자체가 negation ranking과 관련됨을 보고합니다. 

이 \(\Delta_X\)는 사실 당신의 \(d_t\)와 거의 같은 종류의 quantity입니다.

즉,

$$
d_t
\approx
\mathbb E[g(T_-)-g(T_+)]
$$

로 잡으면, [Similarity Is Not Logic]의 직접 negation-direction 분석과 자연스럽게 연결됩니다.

이렇게 하면 논문 간 관계를 더 명확하게 만들 수 있습니다.

* **linear probe \(w_t\)**: “negation information이 분류 가능한가?”
* **difference direction \(d_t\)**: “positive→negative representation이 실제로 어느 방향으로 이동하는가?”
* **\(\gamma\)**: “그 text-side movement와 image-side movement가 실제 similarity interaction으로 얼마나 전환되는가?”

이 세 단계가 서로 다른 질문입니다.

---

## 8. 그래서 현재 논문의 전체 구조를 이렇게 재정의하는 것을 권합니다

현재 논문의 논리를

$$
\text{representation exists}
\rightarrow
\text{direction exists}
\rightarrow
\text{directions aligned}
\rightarrow
\text{interaction dominates}
$$

로 보는 것이 좋습니다.

이를 수식으로 쓰면:

$$
\boxed{
w_i,w_t
\quad\text{measure linear accessibility}
}
$$

$$
\boxed{
d_i,d_t
\quad\text{measure state-transition geometry}
}
$$

$$
\boxed{
\cos(d_i,d_t)
\quad\text{measures cross-modal directional alignment}
}
$$

$$
\boxed{
\gamma
\quad\text{measures the resulting similarity interaction}
}
$$

여기서 가장 중요한 관계는

$$
\gamma
\approx
\frac14 a_i a_t\cos(d_i,d_t)
$$

입니다. 

즉 **\(w\)는 evidence-of-representation이고, \(d\)는 geometry-of-representation이며, \(\gamma\)는 interface-level interaction입니다.**

이렇게 분리하면 기존의 “정보가 있다 vs 없다” 논쟁을 훨씬 깔끔하게 정리할 수 있습니다.

---

### 제 권고

논문에서는 용어를 다음처럼 쓰는 것이 가장 안전합니다.

$$
\boxed{
A_{\mathrm{res}}=\alpha-\gamma
}
$$

“conditional residual affirmation preference”

$$
\boxed{
P_{\mathrm{res}}=\beta-\gamma
}
$$

“conditional residual presence preference”

그리고

$$
\boxed{
d_i=\mu_i^+-\mu_i^-,
\quad
d_t=\mu_t^+-\mu_t^-
}
$$

를 “state-transition directions” 또는 “polarity directions”로 정의합니다.

\(w_i,w_t\)는 “linear probe normals”로 유지하고, **probe normal과 polarity direction을 절대로 같은 것으로 취급하지 않는 것**이 중요합니다.

현재 데이터에서 그 둘이 얼마나 다른지, 즉

$$
\cos(w_i,d_i),\quad
\cos(w_t,d_t),\quad
\cos(d_i,d_t)
$$

를 추가하면 PAPER_4의 geometry argument가 훨씬 견고해질 가능성이 큽니다. 특히 기존 논문이 실제로 “probe-normal alignment는 성공하지 않았고 movement-direction alignment가 필요하다”고 이미 관찰하고 있기 때문에, 이 분석은 자연스럽게 다음 실험으로 이어집니다. 
좋습니다. 그리고 생각보다 더 좋습니다 — 두 양이 추상적 계수가 아니라 **관측된 유사도 두 개의 차이**로 정확히 떨어지기 때문입니다.

## 대수

성공 조건 min(S₊₊,S₋₋) > max(S₊₋,S₋₊)는 부등식 네 개인데, 계수로 쓰면 이렇게 갈립니다.

| 부등식 | 계수 | 관측량 |
|---|---|---|
| S₊₊ > S₊₋ | α + γ > 0 | (S₊₊ − S₊₋)/2 |
| S₊₊ > S₋₊ | β + γ > 0 | (S₊₊ − S₋₊)/2 |
| S₋₋ > S₋₊ | γ − α > 0 | −(S₋₊ − S₋₋)/2 |
| S₋₋ > S₊₋ | γ − β > 0 | −(S₊₋ − S₋₋)/2 |

즉

$$\alpha - \gamma = \tfrac{1}{2}(S_{-+} - S_{--}), \qquad \beta - \gamma = \tfrac{1}{2}(S_{+-} - S_{--})$$

**α − γ**는 "객체가 **없는** 이미지에 대해 긍정 캡션을 부정 캡션보다 얼마나 더 좋아하는가"입니다. 그게 affirmation bias의 정의 그 자체입니다.
**β − γ**는 "**부정** 캡션에 대해 객체가 있는 이미지를 없는 이미지보다 얼마나 더 좋아하는가"입니다. presence preference가 정확한 이름입니다.

이름이 맞을 뿐 아니라, 이 좌표는 논문이 못 하고 있는 문장을 하나 줍니다: **어려움은 전부 S₋₋ 칸에 있습니다.** α+γ, β+γ는 S₊₊가 자기 행·열에서 이기는 조건(쉬움), α−γ, β−γ는 S₋₋가 자기 행·열에서 이기는 조건(실패하는 쪽)입니다. "교차항이 주효과의 1/14"보다 이쪽이 훨씬 잘 읽힙니다.

## 전제 하나를 명시해야 합니다

α ≥ 0, β ≥ 0일 때만 "success ⟺ (α−γ < 0) ∧ (β−γ < 0)"이 성립합니다. 이 부호 하에서는 γ > 0이 성공의 필요조건이므로 α+γ > 0, β+γ > 0이 자동으로 따라오기 때문입니다.

논문은 |α|, |β|만 보고하고 있어서 이 전제가 데이터에서 확인되지 않습니다. 부호 규약(α > 0 = 긍정 편향, β > 0 = 존재 편향)을 §3에서 고정하고 9개 모델의 **부호 붙은 값**을 표 2에 실으십시오. 특히 α 억제로 비율을 옮긴 NegFull 계열(|α| 배율 0.28–0.29×)은 과보정으로 α < 0에 들어갔을 가능성이 있고, 만약 그렇다면 그건 결함이 아니라 §4.4에서 보고할 가치가 큰 결과입니다.

## 주의 네 가지

**1. 자유도가 하나 줍니다.** (α−γ, β−γ) 두 값은 α−β와 α+β−2γ만 결정하므로 γ를 따로 복원할 수 없습니다. §4.2의 순열 검정(γ가 객체 특이적이다)과 §4.4의 경로 분해(α 억제 vs γ 증폭)는 γ를 독립적으로 봐야 성립합니다. **(α, β, γ)를 정식 좌표로 유지하고 (α−γ, β−γ)는 해석용 판독값으로 제시**하십시오. 교체가 아니라 병기입니다.

**2. 스케일 의존적입니다.** γ/max(|α|,|β|)의 장점은 임베딩 스케일 λ에 불변이라는 것인데, α−γ와 β−γ는 λ배 됩니다. 정규화가 필요하면 이렇게 쓸 수 있습니다.

$$r_{\text{text}} = \frac{\alpha - \gamma}{\alpha + \gamma} \in [-1, 1]$$

r = 1은 γ = 0, 즉 텍스트 선호가 이미지 상태와 완전히 무관한 상태(순수 bag-of-words)이고, r = 0이 문턱, r = −1은 α = 0인 이상적 상태입니다. 기준선에 넣으면 r_text = 0.00726/0.00836 = **0.868**, r_image = 0.00409/0.00519 = **0.788**입니다. "완전히 이미지를 무시하는 상태까지 87% 와 있다"는 γ/max = 0.044보다 훨씬 잘 전달됩니다. 둘은 r = (1−γ/α)/(1+γ/α)로 단조 대응하므로 §6의 상관 분석도 그대로 갑니다.

**3. 용어 충돌 가능성.** 문헌에서 affirmation bias는 보통 조건 없는 긍정 선호, 즉 α에 가깝게 쓰입니다. α−γ는 **객체가 없을 때 조건부**로 남는 긍정 선호이므로, 처음 도입할 때 "α는 평균 긍정 선호, α−γ는 객체 부재 조건에서의 긍정 선호이며 실패를 결정하는 것은 후자"라고 한 문장 붙이십시오. 오히려 이게 NegBench류 MCQ가 실제로 재는 양에 더 가깝습니다.

**4. 이건 결국 Winoground의 text/image score입니다.** α−γ < 0의 쌍 단위 비율 = text score, β−γ < 0의 비율 = image score, 둘의 교집합 = group score = 현재 보고 중인 0.88%. 지난 리뷰에서 지적한 지점에 사용자가 대수 쪽에서 독립적으로 도달한 셈입니다. 인용 문제는 여전히 남습니다.

## 그래서 실제로 얻는 것

지금 논문은 **교집합(0.88%)만** 보고하고 어느 쪽이 깨졌는지를 버립니다. 이 분할을 도입하면 세 줄이 추가로 나옵니다.

- 모델별 text score / image score를 나눠 보고 → 실패 축이 보인다.
- §4.3 관찰 3에서 297개 중 75개(25.3%)가 |β| > |α|이고 LAION·SigLIP·NegCLIP은 매크로에서도 그렇다고 썼으니, **|α|>|β| 모델은 text score가, |β|>|α| 모델은 image score가 더 낮아야 한다**는 이중 해리 예측이 나옵니다. 9개 모델로 바로 검증됩니다.
- §4.4의 "억제 경로"가 더 선명해집니다. α를 억누른 세 모델은 정확히 α−γ를 줄인 것이고, 그건 text score만 올리고 image score는 건드리지 않았을 것입니다. NegCLIP의 |β|가 6.11로 기준선보다 큰 것을 보면 실제로 그럴 가능성이 높습니다.

정리하면: **채택하십시오. 단, (α,β,γ)를 대체하지 말고 판독층으로 얹고, 부호 규약을 명시하고, 정규화 형태를 하나 정하십시오.** 이름 자체는 두 개 다 정확합니다.

프레이밍 자체는 옳습니다. 다만 0827의 **느슨한 형태**는 약하고, PAPER의 대수 안에 넣으면 훨씬 강해집니다. 두 형태를 구분해서 답하겠습니다.

## 느슨한 형태가 사주는 것 (실제로 큼)

**1. 비전 인코더를 진단에 강제로 넣습니다.** 이게 가장 큰 값입니다. "부정 이해" 프레이밍은 [1][2][3] 모두 텍스트 쪽으로 끌려갑니다 — 부정은 텍스트의 성질이니까요. 반면 결합은 정의상 양쪽에서 성립해야 하는 관계라서, 프레이밍을 바꾸는 순간 "이미지 인코더도 봐야 한다"가 논증 없이 따라옵니다. 0827의 §1이 정확히 이 일을 하고 있고, 이건 두 버전 통틀어 가장 잘한 수사적 선택입니다.

**2. AB-swap 설계를 생성합니다.** 이게 결정적입니다. "부정 이해" 프레임에서 자연스러운 텍스트 쌍은 "there is a cat" vs "there is no cat"이고, 이건 PAPER §1이 [2]를 비판하는 바로 그 교란(부정 표지 유무만 탐지해도 성립)입니다. 결합 프레임에서 자연스러운 쌍은 "같은 단어 집합, 결합만 교체"이고 그게 AB-swap입니다. **프레이밍이 방법론적 개선을 낳았습니다.** 논문은 이 인과를 명시해야 합니다 — 지금은 프레임과 설계가 따로 서술됩니다.

**3. 우연 이하를 예측합니다.** "CLIP은 no를 모른다"는 우연 근처를 예측하지 8.4~19분의 1을 예측하지 않습니다. bag-of-words 거동은 어느 객체가 부정되었는지가 임베딩을 거의 바꾸지 않는다 → 주효과 지배 → 결정론적 역전을 예측합니다. 관측과 맞는 쪽은 후자입니다.

## 어디서 깨지는가

**속성이 아닙니다.** 고전적 결합 문제는 빨강과 파랑이 **둘 다 이미지에 있고** 어느 것이 어느 객체에 붙는지가 문제입니다. 부정에서 극성은 픽셀에 없습니다. 결합되는 것은 이미지의 *사실*과 텍스트의 *주장*이고, 이건 결합보다 검증(verification/entailment)에 가깝습니다.

여기서 따라오는 실질적 제약이 있습니다: **단일 객체 부정("a photo of no cat")은 결합 문제가 아닙니다.** 결합이 성립하려면 객체가 둘 이상 있어야 하고, 논문의 AB-swap이 그 조건을 인위적으로 만든 것입니다. 즉 이 프레이밍은 **부정 일반이 아니라 논문이 선택한 구성**을 덮습니다. NegBench의 상당 부분이 단일 객체 부정이므로 범위 진술이 필요합니다. "본 연구는 다객체 장면에서의 극성-객체 결합을 다루며, 단일 객체 부정은 이 프레임 밖이다"를 §2나 한계에 넣으십시오. 안 쓰면 리뷰어가 대신 씁니다.

**그리고 데이터가 프레이밍의 처방을 반증합니다.** 결합 프레임의 표준 처방은 [5]의 정렬(텍스트 선형 변환)인데, PAPER는 방향 정렬이 이론값의 절반에서 실패하고 공유 W가 22.99%에서 멈춘다고 보고합니다. **결합 도구상자가 부정에서는 작동하지 않습니다.** 0827은 프레임을 §1에서 선언하고 결론에서 회수하지 않아서, 이 사실이 그냥 손실로 남습니다.

회수하면 강점이 됩니다: *"부정은 결합처럼 보이지만 결합의 처방에 저항한다"* — 이게 논문의 결론이 될 수 있는 문장이고, [5] 대비 차별점이 여기서 나옵니다.

## 강한 형태 — PAPER의 좌표계 안에서

여기서 프레이밍이 은유를 벗고 정의가 됩니다.

$$S_{ab} = C + a\beta + b\alpha + ab\,\gamma$$

- **α, β = bag-of-words 성분.** 점수가 이미지 상태와 텍스트 상태에 각각 따로 반응하는 부분.
- **γ = 결합 성분.** 점수가 두 상태의 *조합*에 반응하는 부분.
- **γ = 0 ⟺ 점수가 가법적 ⟺ 완전한 bag-of-words.**

이건 Yuksekgonul의 "bag of words"와 [5]의 "unimodal에는 있는데 cross-modal에서 잃는다"를 **측정 가능한 양으로 바꾼 것**입니다. 그리고 논문의 결과가 그 언어로 정확히 서술됩니다: γ ≠ 0(결합은 통계적으로 실재, 순열 검정 33개 중 24개) 그러나 γ ≪ max(|α|,|β|)(결합이 결정에 관여하지 못함).

덤으로 [3]과 [5]가 화해합니다. [3]의 "유사도가 연산자와 무관하게 개념 증거를 평균 풀링한다"는 α, β 지배를 말하는 것이고, [5]의 "결합 정보는 있다"는 γ ≠ 0을 말하는 것입니다. 둘은 경쟁 주장이 아니라 **같은 분해의 서로 다른 항**이고, 성공 조건 γ > max(|α|,|β|)가 "결합이 평균 풀링을 이겨야 한다"로 읽힙니다. 관련 연구를 통합하는 이런 문장이 지금 논문에 없습니다.

앞서 정의하신 α−γ, β−γ와도 정합합니다. α−γ < 0은 "텍스트 축에서 결합이 긍정 편향을 이긴다"입니다.

## 권고

- **PAPER §1에는 넣지 마십시오.** 대수는 결합 개념 없이도 자립하고, §1에 은유를 넣으면 명제 1이 해석에 의존하는 것처럼 보입니다.
- **§3 명제 1 직후에 두 문장**으로 넣으십시오: γ = 0이 가법성 = bag-of-words이고, 따라서 γ가 결합의 정량적 정의다.
- **§6에서 [3]·[5] 통합**에 쓰십시오. 여기가 이 프레이밍의 최고 활용처입니다.
- **용어는 "attribute-object binding"보다 "polarity-object binding"** 또는 그냥 "교차항/결합 성분". 전자는 CLEVR·ARO·SugarCrepe 전통의 term of art라서 그 벤치마크들과의 비교를 기대하게 만듭니다.
- **0827의 [4] 인용은 살짝 늘어져 있습니다.** "결합은 두 모달리티 모두에서 성립해야 하므로 이미지 인코더도 진단해야 한다[4]"인데, Koishigarina는 양쪽을 프로빙하되 **고치는 것은 텍스트 쪽**입니다. 근거로 세우려면 "양 모달리티를 프로빙했다"까지만 인용하십시오.

한 줄 요약: **은유로 쓰면 평범하고, γ의 정의로 쓰면 논문의 관련 연구 절이 통째로 좋아집니다.**


가능합니다. 오히려 이 아이디어는 PAPER_4의 논리를 더 명확하게 만들 수 있습니다. 다만 “upper bound”를 두 가지 의미로 구분해야 합니다.

1. **프로브 정확도만 알고 있을 때의 이론적 upper bound**
2. **실제 \(w_i,w_t\)의 binary prediction을 모두 가지고 있을 때의 empirical upper bound**

두 번째가 훨씬 강한 결과입니다.

---

## 1. 문제를 정확히 정의해보겠습니다

이미지 프로브가

$$
\hat a(I)\in\{+1,-1\}
$$

을 출력하고,

텍스트 프로브가

$$
\hat b(T)\in\{+1,-1\}
$$

을 출력한다고 합시다.

여기서

* \(+1\): object present / positive
* \(-1\): object absent / negative

입니다.

이 함수는 원래 CLIP embedding이나 cosine similarity를 전혀 보지 않고, 오직

$$
(\hat a,\hat b)
$$

두 비트만 봅니다.

그러면 가능한 입력은 딱 4개입니다.

$$
(+,+),\quad(+,-),\quad(-,+),\quad(-,-)
$$

따라서 어떤 scoring function \(f\)라도 결국

$$
S_{\rm binary}(I,T)=f(\hat a(I),\hat b(T))
$$

형태입니다.

---

# 2. 가장 자연스러운 함수는 equality score입니다

우리가 원하는 것은

$$
\text{present + positive}
$$

와

$$
\text{absent + negative}
$$

를 높이고,

$$
\text{present + negative}
$$

와

$$
\text{absent + positive}
$$

를 낮추는 것입니다.

그러므로 가장 단순한 함수는

$$
\boxed{
S_{\rm bin}=\hat a\hat b
}
$$

입니다.

그러면

| image | text | \(S_{\rm bin}\) |
| ----- | ---- | --------------: |
| +     | +    |              +1 |
| +     | −    |              −1 |
| −     | +    |              −1 |
| −     | −    |              +1 |

입니다.

즉 사실상

> **“두 프로브의 binary prediction이 서로 일치하면 정답 쪽으로 본다.”**

입니다.

0/1 표현을 쓰면

$$
S_{\rm bin}
=
\mathbf 1[\hat a=\hat b].
$$

---

# 3. 여기서 중요한 사실

이 함수는 **두 프로브가 각각 정확할 필요가 없습니다.**

둘이 똑같이 틀려도 됩니다.

예를 들어 실제 상태가

$$
a=+1,\qquad b=+1
$$

인데 두 프로브가 모두

$$
-1,-1
$$

을 예측했다고 합시다.

두 프로브 모두 틀렸지만,

$$
\hat a\hat b=+1
$$

이므로 matching 관점에서는 올바르게 “positive-positive” 관계를 복원합니다.

따라서 이 upper bound는

$$
\min(\text{image probe acc},\text{text probe acc})
$$

와 같지 않습니다.

이것이 중요한 포인트입니다.

---

# 4. 프로브 정확도만 가지고 upper bound를 계산한다면

이미지 프로브 정확도를 \(p_i\), 텍스트 프로브 정확도를 \(p_t\)라고 합시다.

그리고 우선 가장 단순하게 **두 프로브의 오류가 독립**이라고 가정하겠습니다.

한 상태에서 두 예측이 같은 방향으로 맞을 확률은

$$
q
=
p_i p_t
+
(1-p_i)(1-p_t).
$$

첫 번째 항은 둘 다 맞는 경우,

두 번째 항은 둘 다 틀리는 경우입니다.

2×2 전체가 성공하려면 positive 상태와 negative 상태 모두에서 이 관계가 성립해야 하므로, balanced이고 두 상태에서 정확도가 동일하다고 가정하면

$$
\boxed{
\mathrm{Acc}_{group}^{upper}
=
\left[
p_i p_t+(1-p_i)(1-p_t)
\right]^2
}
$$

가 됩니다.

---

# 5. PAPER_4의 숫자를 넣어보면

논문에서 보고한 이미지 probe 0.630, 텍스트 probe 0.912를 그대로 binary accuracy라고 해석하면,

$$
q
=
0.630\times0.912
+
0.370\times0.088
$$

이므로

$$
q\approx0.607.
$$

따라서

$$
\boxed{
\mathrm{Acc}_{group}^{upper}
\approx 0.607^2
\approx36.9\%
}
$$

입니다.

즉, 아주 거칠게 말하면,

> **두 binary probe만을 사용해서 가장 단순한 equality-based composition을 한다고 해도, 개별 probe가 가진 오류 때문에 약 37% 정도가 자연스러운 upper bound가 될 수 있습니다.**

반면 현재 cosine similarity의 2×2 accuracy는 0.88%입니다. 

이 비교는 상당히 흥미롭습니다.

---

# 6. 하지만 36.9%를 “진짜 upper bound”라고 부르면 안 됩니다

여기가 중요합니다.

위 식은 **독립 오류라는 추가 가정**을 넣은 모델 기반 upper bound입니다.

실제로는 두 probe의 오류가 서로 상관되어 있을 수 있습니다.

오류가 강하게 correlated되어 있으면 오히려 더 높아질 수 있습니다.

예를 들어 두 프로브가 항상 같은 샘플에서 같이 틀린다면, 개별 정확도가 낮더라도 binary composition은 매우 잘 될 수 있습니다.

반대로 오류가 서로 반대로 발생하면 훨씬 나빠집니다.

따라서

$$
p_i,p_t
$$

두 숫자만 가지고는 정확한 upper bound를 정할 수 없습니다.

---

# 7. 정확한 empirical upper bound는 실제 binary predictions로 계산할 수 있습니다

이게 제가 PAPER_4에 실제로 넣고 싶은 분석입니다.

각 2×2 block에 대해 프로브 결과를 모두 저장합니다.

예를 들어

$$
\begin{array}{c|cc}
 & T_+ & T_-\\
\hline
I_+ &(a_1,b_1)&(a_2,b_2)\\
I_- &(a_3,b_3)&(a_4,b_4)
\end{array}
$$

입니다.

각 cell의 입력은 네 가지 중 하나입니다.

$$
(++),(+-),(-+),(--)
$$

그런데 scoring function은 이 **네 입력에 어떤 값을 할당할지만 결정하면 됩니다.**

즉

$$
f_{++},f_{+-},f_{-+},f_{--}
$$

네 숫자만 있으면 됩니다.

따라서 가능한 ranking은 많아야 \(4!=24\)개뿐입니다.

그래서 모든 가능한 함수의 ranking을 brute-force해서

> 가장 높은 2×2 accuracy를 얻는 함수

를 찾을 수 있습니다.

이것이 정말 좋은 의미의 **empirical upper bound**입니다.

---

# 8. 더 엄밀하게 표현하면

각 sample \(k\)에 대해

$$
x_k=(\hat a_k,\hat b_k)
$$

라는 4-valued feature만 존재한다고 합시다.

그리고

$$
y_k\in\{\text{correct},\text{incorrect}\}
$$

를 실제 diagonal/off-diagonal 정답으로 정의합니다.

그러면 문제는

> \(x_k\) 네 가지 값만 보고 correct/incorrect ranking을 얼마나 잘 복원할 수 있는가?

가 됩니다.

이건 사실상 **4-level feature의 Bayes-optimal classifier/ranker** 문제입니다.

따라서 실제 prediction을 가지고 있으면

$$
\boxed{
\text{UB}_{binary}
=
\max_{f:\{++, +-, -+, --\}\rightarrow\mathbb R}
\mathrm{Accuracy}(f)
}
$$

를 직접 계산할 수 있습니다.

그리고 domain이 4개뿐이므로 exhaustive enumeration이 가능합니다.

---

# 9. 여기서 아주 중요한 문제가 하나 있습니다

만약 **같은 binary pattern이 어떤 샘플에서는 정답이고 다른 샘플에서는 오답**이라면, 이 함수는 해결할 수 없습니다.

예를 들어 어떤 샘플에서는

$$
(\hat a,\hat b)=(+,+)
$$

가 correct pairing이고,

다른 샘플에서는 똑같이

$$
(+,+)
$$

인데 wrong pairing이라고 합시다.

그렇다면 \(f(+,+)\)는 두 경우를 동시에 구분할 수 없습니다.

즉 이때 발생하는 error는 **binary probes 자체의 정보 손실 때문에 생기는 irreducible error**입니다.

이것이 아주 좋은 분석 포인트입니다.

---

# 10. 그러면 PAPER_4에서 굉장히 강한 실험이 하나 생깁니다

현재 논문의 주장:

> embedding에는 정보가 있다.

를 더 엄밀하게 바꾸어서 다음 세 단계를 비교할 수 있습니다.

### A. Original cosine

$$
S(I,T)=v^\top t
$$

현재 결과:

$$
2\times2 \text{ accuracy}=0.88\%
$$



### B. Binary probe only

$$
S_{\rm bin}=f(\hat a,\hat b)
$$

그리고 최적 \(f\)의 empirical accuracy를 계산합니다.

### C. Oracle binary labels

프로브가 아니라 실제 ground-truth

$$
a,b
$$

를 넣습니다.

그러면 이론적으로

$$
S_{\rm oracle}=ab
$$

로 거의 100%가 가능합니다.

---

# 11. 이 세 개를 비교하면 엄청 깔끔해집니다

예를 들어 결과가 다음처럼 나온다고 생각해봅시다.

$$
\boxed{
0.88\%
\quad
\rightarrow
\quad
35\%
\quad
\rightarrow
\quad
100\%
}
$$

그렇다면 논문의 메시지가 아주 명확해집니다.

> 원래 cosine similarity의 failure 대부분은 representation에 정보가 없어서가 아니라, 정보가 있더라도 binary composition에 충분히 연결하지 못하기 때문이다.

반대로 binary probe-only upper bound가 예를 들어 5%밖에 안 나온다면 이야기가 달라집니다.

그 경우에는

> “정보는 있다”

라는 기존 probing 결과가 retrieval에 필요한 형태의 정보는 아니라는 쪽으로 해석해야 합니다.

즉 이 실험은 **information availability와 task-relevant compositional information을 직접 분리**합니다.

---

# 12. \(w_i,w_t\)와도 정확히 연결됩니다

당신이 앞에서 말한

$$
w_i,\quad w_t
$$

가 각각 binary probe라면,

$$
\hat a_i=\operatorname{sign}(w_i^\top v)
$$

$$
\hat b_t=\operatorname{sign}(w_t^\top t)
$$

이고,

binary-only scorer는

$$
\boxed{
S_{\rm bin}(v,t)
=
\operatorname{sign}(w_i^\top v)
\operatorname{sign}(w_t^\top t)
}
$$

가 됩니다.

이것은 상당히 중요한 구조입니다.

점수는

$$
S_{\rm bin}
=
\operatorname{sign}(w_i^\top v)
\operatorname{sign}(w_t^\top t)
$$

이므로,

> 이미지에서 “present/absent detector” 하나
> × 텍스트에서 “positive/negative detector” 하나

로 composition을 만들어내는 것입니다.

즉 원래 cosine은

$$
v^\top t
$$

였지만,

binary probe composition은

$$
\operatorname{sign}(w_i^\top v)
\operatorname{sign}(w_t^\top t)
$$

입니다.

이 차이를 논문에서 보여주면 꽤 의미가 있습니다.

---

# 13. 그리고 \(d_i,d_t\)와도 연결할 수 있습니다

만약 \(w_i\)가 \(d_i\)와 잘 aligned되어 있고,

$$
w_t
$$

도

$$
d_t
$$

와 잘 aligned되어 있다면,

binary probe가 단순한 arbitrary classifier가 아니라 실제 state-transition direction을 읽고 있다고 해석할 근거가 생깁니다.

따라서 제가 추천하는 전체 chain은:

$$
\boxed{
d_i,d_t
\rightarrow
w_i,w_t
\rightarrow
(\hat a,\hat b)
\rightarrow
S_{\rm bin}
\rightarrow
2\times2\ accuracy
}
$$

입니다.

그리고 original CLIP은

$$
v^\top t
$$

를 바로 사용합니다.

이렇게 놓으면 논문의 핵심 질문은

> **representation에 존재하는 binary information이 왜 original bilinear similarity에서는 composition으로 구현되지 않는가?**

로 매우 깔끔해집니다.

---

# 14. 다만 한 가지 더 조심해야 합니다

“두 binary probe만 보면 가능한 upper bound”에는 두 종류가 있습니다.

**이론적 upper bound**

$$
\left[p_ip_t+(1-p_i)(1-p_t)\right]^2
$$

→ 독립 오류 등의 가정이 필요합니다.

**실증적 upper bound**

$$
\max_f \mathrm{Acc}(f(\hat a,\hat b))
$$

→ 실제 probe prediction을 이용해 직접 계산합니다.

저는 **논문에는 두 번째만 넣는 것을 권합니다.**

첫 번째는 supplementary에서 “독립 오류를 가정한 reference calculation” 정도로 두면 됩니다.

---

## 그리고 한 단계 더 나가면

사실 더 좋은 분석은 binary 출력 자체뿐 아니라 **probe의 signed margin**

$$
m_i=w_i^\top v,\qquad
m_t=w_t^\top t
$$

두 개만 사용하는 경우입니다.

그러면

$$
S=m_i m_t
$$

라는 아주 작은 bilinear scorer를 만들 수 있습니다.

이것은 당신 논문의 rank-1 bilinear

$$
W=ab^\top
$$

와 거의 정확히 연결됩니다.

즉,

$$
v^\top(ab^\top)t
=
(a^\top v)(b^\top t).
$$

따라서

$$
\boxed{
S=(w_i^\top v)(w_t^\top t)
}
$$

는 **“이미지 binary detector × 텍스트 binary detector”라는 가장 작은 cross-dimensional interaction**입니다.

이게 아주 흥미롭습니다. 현재 논문의 rank-1 실험이 22.11%까지 올라갔는데도 완전히 입증되지 않았던 이유를, “rank-1 bilinear이 아무렇게나 학습된 것이 아니라 실제 unimodal probe directions의 outer product인가?”라는 방식으로 재구성할 수 있습니다. 현재 논문은 rank-1 \(W=ab^\top\)를 이러한 detector product로 해석하고 있습니다. 

제가 보기에는 **2페이지 버전에서는 이 binary-probe upper bound 실험이 기존의 9-model/intervention 실험보다 훨씬 가치가 높습니다.** 왜냐하면 “정보가 있다 → 하지만 similarity가 못 쓴다”라는 논문의 핵심 주장을 직접 시험하기 때문입니다.


네. 2페이지라면 지금 논문을 “축약”해서는 안 되고, 사실상 논문의 질문을 하나로 다시 잡는 것이 좋습니다.

제가 재구성한다면 핵심 질문을 다음으로 좁히겠습니다.

> **CLIP은 객체의 존재/부재와 문장의 긍정/부정 정보를 어느 정도 표현하고 있는데, 왜 원래의 cosine similarity에서는 이 정보가 올바른 매칭으로 연결되지 않는가?**

그리고 답을 딱 세 단계로 보여줍니다.

$$
\text{information exists}
\rightarrow
\text{but main effects dominate}
\rightarrow
\text{therefore cross-modal interaction is too weak}
$$

이렇게 가면 2페이지에서도 논문 하나의 메시지가 생깁니다.

현재 PAPER_4는 33개 개념, 1,357개 최소쌍, 9개 모델까지 상당히 많은 실험을 하고 있습니다.  하지만 2페이지에서는 이것을 모두 보여주면 오히려 핵심이 흐려집니다.

제가 쓴다면 다음 구조로 갑니다.

---

# 1. 제목부터 바꾸겠습니다

현재의 제목보다 조금 더 좁고 직접적으로 가는 게 좋습니다.

예를 들어:

> **CLIP의 부정 매칭 실패: 표현 정보와 유사도 상호작용의 괴리**

또는 조금 더 학술적으로:

> **CLIP Negation Failure through the Lens of Main Effects and Cross-modal Interaction**

국내 학술대회라면 첫 번째가 더 낫습니다.

중요한 것은 제목에서 “negation understanding의 근본 원인을 밝혔다”처럼 과장하지 않는 것입니다.

---

# 2. 2페이지의 전체 논리

페이지를 다음처럼 나누겠습니다.

### 1페이지

문제 → 기존 설명의 허점 → 제안하는 2×2 분석 → 데이터

### 2페이지

핵심 결과 → probing/geometry → 논의 → 결론

그리고 **LABCLIP, steering, rank-1/2 bilinear, 9개 모델의 상세 결과는 전부 삭제하거나 한 문장으로 처리**하겠습니다.

현재 논문에는 실제로 여러 개입 방법이 들어가 있지만, 2페이지 논문에서 이것들을 모두 설명하면 “무엇을 발견했는지”가 아니라 “여러 실험을 해봤다”는 인상을 줍니다.

---

# 3. Section 1. 서론 — 약 1/3페이지

여기서는 기존 연구를 네 문장 정도로 끝냅니다.

논문의 출발점은 다음입니다.

CLIP과 같은 dual-encoder VLM은 이미지와 텍스트를 각각 하나의 벡터로 변환하고 cosine similarity로 매칭한다. 이 방식은 일반적인 image-text matching에는 강하지만, “no dog”처럼 객체의 부재를 요구하는 질의에서는 반복적으로 실패한다. 기존 연구는 이 실패를 크게 세 가지 관점에서 설명해왔다. 즉, representation에 negation 정보가 존재하는지에 대한 probing, embedding geometry에 존재하는 negation direction, 그리고 similarity 대신 외부에서 논리 연산을 수행하는 방법이다. 

그런데 여기서 문제를 제기합니다.

> **“정보가 embedding에 존재한다”는 것은 “그 정보가 원래 similarity의 ranking을 결정할 수 있다”는 것을 의미하지 않는다.**

이 한 문장이 사실상 introduction의 핵심입니다.

[Similarity Is Not Logic]도 evidence extraction과 constraint execution을 분리해야 한다고 주장합니다. 

따라서 우리는 다음을 묻습니다.

> **표현된 정보와 실제 similarity ranking 사이에는 어떤 정량적 차이가 존재하는가?**

---

# 4. Section 2. 2×2 분석 — 약 1/2페이지

여기가 논문의 이론적 핵심입니다.

객체 \(X\)가 있는 이미지/없는 이미지와, \(X\)를 긍정/부정하는 두 문장을 조합합니다.

|         |   Positive |   Negative |
| ------- | ---------: | ---------: |
| Present | \(S_{++}\) | \(S_{+-}\) |
| Absent  | \(S_{-+}\) | \(S_{--}\) |

이 네 점수를

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

로 분해합니다.

그리고 각 계수를 한 문장으로 설명합니다.

* \(\alpha\): 텍스트가 positive인지 negative인지에 따른 **text main effect**
* \(\beta\): 이미지에 객체가 있는지 없는지에 따른 **image main effect**
* \(\gamma\): 이미지 상태와 텍스트 상태가 함께 작용하는 **cross-modal interaction**

현재 논문의 가장 좋은 부분은 바로 이것입니다. 

그리고 딱 하나의 proposition만 보여줍니다.

$$
\boxed{
\text{correct matching}
\iff
\gamma>\max(|\alpha|,|\beta|)
}
$$

즉,

> **논리적으로 맞는 이미지-문장 조합을 고르려면 interaction이 각각의 단순한 선호보다 강해야 한다.**

이것으로 충분합니다.

현재 논문은 더 나아가 \(\gamma>0\)인 쌍의 비율 등을 분석하지만, 2페이지에서는 빼겠습니다. 특히 \(\gamma>0\) 자체는 Winoground/GroupMatch 계열의 판정과 연결되는 것이므로 새로운 평가법처럼 보이면 안 됩니다. 논문도 이를 명시적으로 인정하고 있습니다. 

---

# 5. 데이터셋 설명은 매우 짧게

여기서 BEAF + AB-swap만 강조합니다.

이미지:

> COCO 기반 BEAF의 동일 장면 object-removal minimal pairs
> 33 concepts / 1,357 pairs

텍스트:

> AB-swap 문장쌍
> 동일한 단어 집합을 유지하면서 어떤 객체가 부정되는지만 교체

현재 데이터 설계에서 텍스트 pair의 94.8%가 동일 token multiset을 갖고, 99.1%가 동일 token length를 가집니다. 따라서 단순히 “not이라는 단어가 있네”를 탐지하는 shortcut을 상당히 억제합니다. 

여기서 Figure 1을 크게 하나 넣습니다.

**Figure 1. Controlled 2×2 setup**

왼쪽:

사진 A: object present
사진 B: object absent

오른쪽:

“a truck”
“no truck”

가운데 네 개의 \(S\)를 배치하고 아래에 \(\alpha,\beta,\gamma\)를 보여줍니다.

2페이지 논문에서는 이 그림 하나가 사실상 이론 + 데이터 설명을 동시에 담당해야 합니다.

---

# 6. Section 3. 핵심 결과 — 2페이지의 중심

여기서 결과를 딱 세 개만 남기겠습니다.

## Result 1. 정보는 있지만 similarity는 실패한다

기준 CLIP에서:

$$
\text{image probe AUC}=0.759
$$

$$
\text{text probe AUC}=0.912
$$

그런데 2×2 matching:

$$
0.88\%
$$

입니다. 

이 세 숫자를 한 Figure 또는 Table로 보여주면 굉장히 강합니다.

그리고 이렇게 해석합니다.

> 객체의 존재/부재와 polarity는 embedding에서 어느 정도 선형적으로 복원되지만, 원래 cosine similarity를 사용한 조합에서는 거의 활용되지 않는다.

이것이 기존 probing 연구와의 차별점입니다.

---

# 7. Result 2. 왜 실패하는가? → α, β가 γ를 압도한다

기준 CLIP에서

$$
|\alpha|=0.00781
$$

$$
|\beta|=0.00464
$$

$$
\gamma=0.00055
$$

입니다. 

따라서

$$
\frac{\gamma}{\max(|\alpha|,|\beta|)}
=0.0439.
$$

즉 필요한 interaction의 약 4.4%밖에 없습니다.

이것이 사실상 논문의 **main result**입니다.

그리고 아주 중요한 표현을 사용합니다.

> CLIP does not simply “lack negation information”; rather, the interaction required to use that information is overwhelmed by modality-specific main effects.

이 문장이 논문의 중심 문장이 되어야 합니다.

---

# 8. 여기서 당신이 고민한 α−γ, β−γ를 넣을 수 있습니다

이 부분은 오히려 2페이지 논문에서 상당히 유용합니다.

다음처럼 정의합니다.

$$
A_{\mathrm{res}}
=
\alpha-\gamma
=
\frac{S_{-+}-S_{--}}{2}
$$

$$
P_{\mathrm{res}}
=
\beta-\gamma
=
\frac{S_{+-}-S_{--}}{2}.
$$

각각

> residual affirmation preference

> residual presence preference

라고 부릅니다.

이것은 단순한 \(\alpha,\beta\)보다 직관적입니다.

왜냐하면 이것은 실제로

> **객체가 없는 이미지에서도 positive caption을 더 좋아하는가?**

와

> **negative caption을 보면서도 객체가 있는 이미지를 더 좋아하는가?**

를 직접 나타내기 때문입니다.

즉 2×2 표에 다시 연결할 수 있습니다.

저라면 Figure 1에서 이걸 함께 보여줍니다.

---

# 9. Result 3. \(w\)와 \(d\)를 넣어서 “정보 → 방향 → interaction” 구조를 만듭니다

이 부분이 제가 새로 추가하고 싶은 실험입니다.

현재 논문은 probing과 geometry를 따로 이야기합니다. 그런데 2페이지에서는 이것들을 하나의 chain으로 연결할 수 있습니다.

이미지 embedding에 linear probe를 학습합니다.

$$
w_i
$$

텍스트 embedding에도 probe를 학습합니다.

$$
w_t
$$

이것은

> “어떤 방향으로 보면 present/absent 또는 positive/negative를 분류할 수 있는가?”

입니다.

그런데 실제 데이터의 평균 이동 방향도 계산합니다.

$$
d_i
=
\mu_{present}-\mu_{absent}
$$

$$
d_t
=
\mu_{positive}-\mu_{negative}.
$$

이것은

> “실제로 상태가 바뀔 때 embedding이 어느 방향으로 움직이는가?”

입니다.

그러면 세 가지 quantity를 비교합니다.

$$
\cos(w_i,d_i)
$$

$$
\cos(w_t,d_t)
$$

$$
\cos(d_i,d_t).
$$

여기서 가장 중요한 것은 마지막 것입니다.

$$
\boxed{\cos(d_i,d_t)}
$$

이것이 cross-modal polarity direction alignment입니다.

그리고 기존 논문의 local approximation을 연결합니다.

$$
\gamma
\approx
\frac14 a_i a_t
\cos(d_i,d_t).
$$

현재 논문도 이런 형태의 geometry 해석을 이미 사용하고 있습니다. 

이렇게 하면 논문 전체가 훨씬 예뻐집니다.

---

# 10. 그러면 논문의 스토리가 이렇게 됩니다

### Level 1 — Representation

$$
w_i,w_t
$$

를 이용한 probe가 잘 된다.

→ **정보가 있다.**

### Level 2 — Geometry

$$
d_i,d_t
$$

를 측정한다.

→ **그 정보가 실제 embedding 공간에서 특정 방향으로 나타난다.**

### Level 3 — Cross-modal alignment

$$
\cos(d_i,d_t)
$$

를 본다.

→ **두 modality의 방향이 얼마나 맞는가?**

### Level 4 — Similarity

$$
\gamma
$$

를 본다.

→ **그 alignment가 실제 matching score의 interaction으로 얼마나 전환되는가?**

### Level 5 — Retrieval

$$
\gamma>\max(|\alpha|,|\beta|)
$$

가 되어야 실제 ranking이 뒤집힌다.

이게 2페이지 논문의 가장 좋은 구조라고 생각합니다.

---

# 11. 대신 기존 실험 중 과감하게 버릴 것

2페이지라면 저는 다음을 삭제합니다.

### ① 9개 모델 전체 분석

현재 논문에서는 297개 concept-model 조합에서 성공 조건을 만족하는 것이 하나도 없다는 결과가 있습니다. 

좋은 결과지만 2페이지에서는 baseline + 2개 정도만 보여주는 게 낫습니다.

예:

* OpenAI CLIP
* SigLIP
* NegCLIP 또는 CoN-CLIP

그리고 나머지는

> “The same pattern was observed across nine models.”

한 문장으로 끝냅니다.

---

### ② LABCLIP 실험

현재 LABCLIP은 in-sample 82.64%에서 holdout 11.57%로 무너지는 흥미로운 결과가 있습니다.

하지만 2페이지에서는 너무 많은 질문을 불러옵니다.

> 왜 LABCLIP인가?
> 어떤 W인가?
> 왜 train/test split인가?
> 왜 overfitting인가?

이것만 설명하다가 논문의 핵심을 잃습니다.

삭제하는 게 낫습니다.

---

### ③ rank-1/rank-2 bilinear

22.99%라는 결과는 재미있지만, 논문의 핵심 메시지를 오히려 흐립니다. 게다가 통계적으로도 강한 결과가 아닙니다. 따라서 본문에서 빼겠습니다.

---

### ④ steering experiment

현재 논문의 steering 결과는 geometry 논문과 비교하기에는 좋지만, 2페이지에서는 너무 많은 가정이 들어갑니다.

> probe direction인가?
> movement direction인가?
> rotation인가?
> scaling인가?

이것도 삭제합니다.

---

### ⑤ “negation fine-tuning이 α를 움직인다” 분석

흥미롭지만 이것은 별도의 논문이 될 수 있는 내용입니다.

현재 논문의 중심 질문에는 필수적이지 않습니다.

---

# 12. 대신 꼭 추가하고 싶은 실험

딱 하나를 추가한다면 저는 **\(w\) vs \(d\) 분석**을 합니다.

왜냐하면 이것이 지금 논문에서 가장 약한 연결고리를 메워주기 때문입니다.

현재 논문의 암묵적 논리는

$$
\text{probe}
\rightarrow
\text{information}
\rightarrow
\text{direction}
\rightarrow
\gamma
$$

인데 중간 단계가 충분히 검증되지 않았습니다.

따라서 다음 표 하나면 됩니다.

| Model   | AUC\(_i\) | AUC\(_t\) | cos(\(w_i,d_i\)) | cos(\(w_t,d_t\)) | cos(\(d_i,d_t\)) | \(\gamma\) |
| ------- | --------: | --------: | ---------------: | ---------------: | ---------------: | ---------: |
| CLIP    |           |           |                  |                  |                  |            |
| SigLIP  |           |           |                  |                  |                  |            |
| NegCLIP |           |           |                  |                  |                  |            |

그리고 가능하면 concept별로

$$
\gamma_k
\quad\text{vs}\quad
\cos(d_{i,k},d_{t,k})
$$

상관을 봅니다.

이게 의미 있게 나온다면 논문의 기여가 단순한 “2×2 score decomposition”에서 한 단계 올라갑니다.

---

# 13. 아주 중요한 주의점

다만 이 실험에서 \(w_i,w_t\)를 직접 cross-modal cosine으로 비교하면 안 됩니다.

이미지와 텍스트의 embedding coordinate가 같은 CLIP shared space에 있더라도, probe weight 자체는 각 modality의 covariance에 영향을 받습니다.

즉

$$
w_i\not\equiv d_i
$$

이고,

$$
w_t\not\equiv d_t
$$

입니다.

더 중요한 것은

$$
\cos(w_i,w_t)
$$

보다

$$
\cos(d_i,d_t)
$$

가 현재 논문의 \(\gamma\)와 더 직접적인 관계를 갖는다는 것입니다.

이 점을 논문에서 명시하면 상당히 깔끔합니다.

---

# 14. 2페이지에서 Figure는 딱 2개

저라면 표를 여러 개 만들지 않고 그림 2개를 만듭니다.

### Figure 1. Conceptual framework

왼쪽:

**Present / Absent image**

×

**Positive / Negative text**

↓

2×2 similarity matrix

↓

$$
\alpha,\beta,\gamma
$$

↓

$$
\gamma>\max(|\alpha|,|\beta|)
$$

이 그림 하나로 Section 1~2를 설명합니다.

---

### Figure 2. Main empirical result

x축:

$$
\max(|\alpha|,|\beta|)
$$

y축:

$$
\gamma
$$

그리고 diagonal

$$
y=x
$$

을 그립니다.

모든 모델/개념이 거의 아래쪽에 위치하도록 합니다.

그 옆에 작은 inset 또는 막대를 넣어서

$$
\text{probe AUC}
\quad\text{vs}\quad
\text{2×2 accuracy}
$$

를 보여줍니다.

그러면 그림 하나로

> **정보는 있다 → 그런데 interaction은 약하다**

가 시각적으로 끝납니다.

---

# 15. 최종적으로 2페이지의 구성은 이렇게 하겠습니다

## 제목

**CLIP의 부정 매칭 실패: 표현 정보와 유사도 상호작용의 괴리**

### 1. 서론 — 0.3p

* CLIP은 negation에서 실패
* 기존 연구: 정보는 embedding에 존재한다고 주장
* 그러나 representation availability ≠ similarity usability
* 연구 질문 제시

### 2. 방법 — 0.5p

* BEAF minimal pair
* AB-swap text pair
* 2×2 similarity matrix
* \(\alpha,\beta,\gamma\) decomposition
* success condition

$$
\gamma>\max(|\alpha|,|\beta|)
$$

### 3. 결과 — 0.8p

**3.1 정보는 존재한다**

$$
AUC_i=0.759,\quad AUC_t=0.912
$$

**3.2 그러나 interaction은 너무 작다**

$$
|\alpha|=0.00781,\quad
|\beta|=0.00464,\quad
\gamma=0.00055
$$

$$
\gamma/\max(|\alpha|,|\beta|)
=0.0439
$$

**3.3 representation geometry**

$$
w_i,w_t
$$

vs

$$
d_i,d_t
$$

그리고

$$
\cos(d_i,d_t)
$$

와 \(\gamma\)의 관계.

### 4. 논의 및 결론 — 0.3~0.4p

핵심 결론:

> CLIP의 negation failure는 단순한 정보 부재로 설명하기 어렵다. 객체 존재/부재와 문장 polarity는 embedding에서 선형적으로 복원되지만, 원래 similarity에서는 modality-specific main effects가 cross-modal interaction을 압도한다.

그리고 마지막에 딱 한계 하나만 씁니다.

> 본 연구는 object-presence negation에 초점을 맞추므로 relation/action negation으로의 일반화는 검증하지 않았다.

이 정도면 충분합니다.

---

# 16. 참고문헌도 5~6개만 남깁니다

2페이지에서 참고문헌을 많이 넣는 것은 손해입니다.

저라면:

1. Winoground
2. Vision-Language Models Do Not Understand Negation
3. CLIP Behaves like a Bag-of-Words
4. Similarity Is Not Logic
5. When Negation Is a Geometry Problem
6. BEAF

정도만 남깁니다.

특히 [Similarity Is Not Logic]은 매우 중요합니다. 그 논문 역시 “negation signal 자체는 embedding에 존재하지만 holistic similarity가 이를 활용하지 못한다”고 직접 분석하고 있습니다. 텍스트의 \(g(\text{“not X”})-g(\text{“X”})\) 방향도 실제로 측정했습니다. 

따라서 PAPER_4의 novelty는

> “negation 정보가 존재한다”

가 될 수 없습니다.

오히려

> **“그 정보가 존재하는데도 2×2 similarity에서 왜 실패하는지를 main-effect/interaction 관점에서 정량화한다.”**

로 잡아야 합니다.

---

## 제가 가장 추천하는 최종 논리

사실 이 논문은 다음 네 문장만 살아 있으면 됩니다.

> **① CLIP embedding에는 polarity와 object-presence 정보가 존재한다.**

$$
\text{probe AUC}\gg 50\%
$$

> **② 그러나 이 정보는 원래 similarity에서 올바른 matching으로 충분히 전환되지 않는다.**

$$
\text{2×2 accuracy}\ll\text{chance}
$$

> **③ 네 개의 similarity score를 분해하면 modality-specific main effects가 cross-modal interaction보다 훨씬 크다.**

$$
\max(|\alpha|,|\beta|)\gg\gamma
$$

> **④ 따라서 negation matching의 병목은 단순한 representation absence라기보다, 표현된 정보를 cross-modal similarity interaction으로 변환하는 능력에 있다.**

이렇게 만들면 기존의 “CLIP이 negation을 이해하지 못한다”는 다소 넓은 주장을 피하면서도, [Winoground]의 compositionality 문제, [CLIP BoW]의 cross-modal binding 문제, [Similarity Is Not Logic]의 interface 문제, [Geometry]의 direction 문제를 하나의 작은 프레임으로 연결할 수 있습니다.  

그리고 **2페이지 논문으로서는 지금보다 오히려 이 버전이 더 강합니다.** 현재 PAPER_4의 여러 intervention 결과를 전부 유지하면 “무엇을 발견했는가”가 희석되지만, `probe → direction → interaction`이라는 하나의 축으로 재구성하면 연구 질문과 결과가 훨씬 선명해집니다.

원한다면 다음 단계로는 이 구조를 그대로 사용해서 **실제 국내 학술대회 2페이지 형식의 초록 + 본문 + Figure 2개에 들어갈 문구까지 포함한 초안**으로 재작성하는 것이 가장 좋습니다.
