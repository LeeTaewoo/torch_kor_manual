<a name='optim.dok'></a>
# Optim 패키지

이 패키지는 일련의 최적화 알고리즘들을 제공합니다. 
그 알고리즘들은 모두 통일된 closure(함수 안에 정의된 함수) 기반 API를 따릅니다.

이 패키지는 [nn](http://nn.readthedocs.org) 패키지와 완전히 호환됩니다.
또한, 이 패키지는 임의의 목적 함수들을 최적화하기 위해 사용될 수도 있습니다.

현재는, 다음 알고리즘들이 제공됩니다:

  * [통계적 기울기 하강(Stochastic Gradient Descent)](#optim.sgd)
  * [평균된 통계적 기울기 하강(Averaged Stochastic Gradient Descent)](#optim.asgd)
  * [L-BFGS](#optim.lbfgs)
  * [Conjugate Gradients](#optim.cg)

이 모든 알고리즘은 통계적 최적화뿐만 아니라 배치(batch) 최적화도 지원하기 
위해 설계되었습니다. 평가(evaluation)는 한 함수에 인자를 대입하여 그 함수의 
결괏값을 얻는 연산입니다. 목적(objective)은 한 목적 함수를 평가한 값입니다.
배치, 미니 배치, 또는 한 단일 샘플에서 그 목적을 평가하기 위한 한 목적 함수가
있다고 할 때, 통계적 최적화를 지원할지 또는 배치 최적화를 지원할지는 
그 목적 함수를 만드는 사용자에게 달려 있습니다.

이 알고리즘 중 몇몇은 line search를 지원합니다. line search는 한 함수(L-BFGS)로
전달될 수 있습니다. 다른 함수들은 오직 학습률만 지원합니다 (SGD).

<a name='optim.overview'></a>
## 개요

이 패키지는 [Torch](https://github.com/torch/torch7/blob/master/README.md)를 위한 몇 개의 최적화 루틴들을 담고 있습니다.
각 최적화 알고리즘은 같은 인터페이스를 가집니다:

```lua
x*, {f}, ... = optim.method(func, x, state)
```

여기서:

* `func`: 사용자가 정의한 함수안에 정의된 함수(closure)입니다. 다음 API를 따릅니다: `f, df/dx = func(x)`
* `x`: 현재 파라미터 벡터 (1차원 `torch.Tensor`)
* `state`: 파라미터들과 알고리즘 의존적 상태 변수들로 구성된 테이블 하나
* `x*`: `f, x* = argmin_x f(x)`를 최소화하는 새 파라미터 벡터
* `{f}`: 모든 f 값들의 테이블 하나, 그것들이 평가된 순서대로 (SGD 같은 몇몇 단순한 알고리즘의 경우, `#f == 1`)

<a name='optim.example'></a>
## 예

상태 테이블은 그 알고리즘의 상태를 들고 있기 위해 사용됩니다.
그 테이블은 보통 사용자에 의해 한 번만 초기화됩니다. 그리고 optim 함수로 블랙 박스처럼 전달됩니다. 예:


```lua
state = {
   learningRate = 1e-3,
   momentum = 0.5
}

for i,sample in ipairs(training_samples) do
    local func = function(x)
       -- 평가(eval) 함수 정의
       return f,df_dx
    end
    optim.sgd(func,x,state)
end
```

<a name='optim.algorithms'></a>
## 알고리즘

제공되는 모든 알고리즘은 통일된 인터페이스를 가집니다:
```lua
w_new,fs = optim.method(func,w,state)
```
여기서: 
w는 훈련할 수 있거나 조절할 수 있는 파라미터 벡터입니다.
상태는 그 알고리즘을 위한 옵션들과 그 알고리즘의 상태를 담고 있습니다.
func는 다음 인터페이스를 가진 함수 안에 정의된 함수입니다.
```lua
f,df_dw = func(w)
```
w_new는 새 파라미터 벡터입니다 (최적화 후에),
fs는 최적화 과정 동안 평가된 그 objective의 모든 값들을 포함하는 한 테이블입니다.
: fs[1]은 최적화 전의 값입니다, 그리고 fs[#fs]는 가장 최적화된 것입니다 (가장 작은 것).

<a name='optim.sgd'></a>
### [x] sgd(func, w, state) 

통계적 기울기 하강(SGD)의 구현.

인자:

  * `opfunc` : 평가될 시점의, 한 단일 입력 (`X`)을 받는 함수 하나, 그리고 `f(X)`와 `df/dX`를 리턴.
  * `x`      : 시작 점
  * `config` : 그 최적화기를 위한 설정 파라미터들을 가진 테이블
  * `config.learningRate`      : 학습률
  * `config.learningRateDecay` : 학습률 쇠퇴
  * `config.weightDecay`       : 가중치 쇠퇴
  * `config.weightDecays`      : 개별 가중치 쇠퇴로 구성된 벡터
  * `config.momentum`          : 모멘텀
  * `config.dampening`         : 모멘텀을 위한 감쇠(dampening)
  * `config.nesterov`          : Nesterov 모멘텀을 사용할 수 있게 함
  * `state`  : 그 최적화기의 상태를 설명하는 한 테이블; 상태가 수정된 각 호출 뒤에
  * `state.learningRates`      : 개별 학습률들로 구성된 벡터

리턴:

  * `x`     : 새 x 벡터
  * `f(x)`  : 갱신 전에 평가된 함수

<a name='optim.asgd'></a>
### [x] asgd(func, w, state) 

평균된 통계적 기울기 하강(ASGD)의 구현:

```
x = (1 - lambda eta_t) x - eta_t df/dx(z,x)
a = a + mu_t [ x - a ]

eta_t = eta0 / (1 + lambda eta0 t) ^ 0.75
mu_t = 1/max(1,t-t0)
```

인자:

  * `opfunc` : 평가될 시점의, 한 단일 입력 (`X`)을 받는 함수 하나, 그리고 `f(X)`와 `df/dX`를 리턴.
  * `x` : 시작 점
  * `state` : 그 최적화기의 상태를 설명하는 한 테이블; 상태가 수정된 각 호출 뒤에
  * `state.eta0` : 학습률
  * `state.lambda` : 쇠퇴 항
  * `state.alpha` : 에타(eta) 갱신을 위한 파워(power, 제곱 형태로 곱해지는 항인듯 합니다)
  * `state.t0` : 평균내기를 시작하는 지점

리턴:

  * `x`     : 새 x 벡터
  * `f(x)`  : 갱신 전에 평가된 함수
  * `ax`    : 평균된 x 벡터


<a name='optim.lbfgs'></a>
### [x] lbfgs(func, w, state)

사용자가 제공한 line search 함수(`state.lineSearch`)에 의존하는 L-BFGS의 구현.
만약 이 함수가 제공되지 않으면, 고정된 크기 스텝들을 만들기 위해 단순한 학습률이 사용됩니다.
고정된 크기 스텝들은 line search보다 훨씬 비용이 덜 듭니다.
그리고 통계적 문제들에 유용할 수 있습니다.

학습률은 line search가 제공될 때에도 사용됩니다.
학습률은 또한 대규모 통계적 문제들에도 유용합니다.
여기서 opfunc는 `f(x)`의 한 잡음 있는 근사치입니다.
그 경우, 학습률은 그 스텝 크기에서의 신뢰 감소를 허용합니다.

인자:

  * `opfunc` : 평가될 시점의, 한 단일 입력 (`X`)을 받는 함수 하나, 그리고 `f(X)`와 `df/dX`를 리턴.
  * `x` : 시작 점
  * `state` : 그 최적화기의 상태를 설명하는 한 테이블; 상태가 수정된 각 호출 뒤에
  * `state.maxIter` : 몇 번까지 반복할 지를 나타내는 최대 횟수
  * `state.maxEval` : 함수 평가의 최대 횟수
  * `state.tolFun` : 일차 최적(optimality)의 종료 허용치
  * `state.tolX` : func/param 변화의 측면에서, 진행 과정에서 종료 허용치
  * `state.lineSearch` : line search 함수
  * `state.learningRate` : 만약 line search가 제공되지 않으면, 한 고정된 스텝 크기가 사용됨

리턴:
  * `x*` : 그 최적 포인트에서의 새 `x` 벡터
  * `f`  : 모든 함수 값들로 구성된 테이블: 
   * `f[1]`는 어떤 최적화도 되기 전의 함수 값, 그리고
   * `f[#f]`는 `x*`에서의 완전히 최적화된 마지막 값


<a name='optim.cg'></a>
### [x] cg(func, w, state)

Conjugate Gradient 방법의 한 구현. 이 함수는 Carl E. Rasmussen가 작성한
`minimize.m`를 다시 쓴 것입니다. 이 함수는 정확히 같은 결과들을 만든다고
가정됩니다 (주거나 받는, 몇몇 바뀐 연산 순서로 인한 수치적 정확도 ).
우리는 rosenbrock에 대한 결과를 [minimize.m](http://www.gatsby.ucl.ac.uk/~edward/code/minimize/example.html)와 비교할 수 있습니다.
```
[x fx c] = minimize([0 0]', 'rosenbrock', -25)
```

유념하십시오. 우리는 오직 함수 평가의 횟수만 제한합니다, 
그것은 실제 사용에서 훨씬 더 중요합니다.

인자:

  * `opfunc` : 평가의 포인트인, 한 단일 입력을 받는 함수 하나.
  * `x`      : 시작 점
  * `state` : 파라미터들과 일시적 할당들로 구성된 테이블
  * `state.maxEval`     : 함수 평가의 최대 횟수
  * `state.maxIter`     : 반복의 최대 횟수
  * `state.df[0,1,2,3]` : 만약 torch.Tensor가 전달되면, 그것은 임시 스토리지로 사용될 것입니다.
  * `state.[s,x0]`      : 만약 torch.Tensor가 전달되면, 그것은 임시 스토리지로 사용될 것입니다.

리턴:

  * `x*` : 그 최적 포인트에서의 새 x 벡터
  * `f`  : 모든 함수 값들로 구성된 테이블: 
   * `f[1]`는 어떤 최적화도 되기 전의 함수 값, 그리고
   * `f[#f]`는 `x*`에서의 완전히 최적화된 마지막 값


