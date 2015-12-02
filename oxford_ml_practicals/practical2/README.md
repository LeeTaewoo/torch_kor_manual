# 실습 2
기계 학습, 2015년 봄

## 설정
설정은 [실습 1](https://github.com/oxford-cs-ml-2015/practical1)을 참고해 주십시오.

# 실습을 위한 코스 페이지
<https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/>

# 실습
실습 2를 내려받습니다.
 
```bash
git clone https://github.com/LeeTaewoo/demos.git ~/torchDemos --recursive
```

실행해봅니다.

```bash
th ./example-linear-regression.lua
```

```bash
current loss = 1.5824611014963
current loss = 1.5824378295574	
id  approx   text	
 1   40.09  40.32	
 2   42.76  42.92	
 3   45.21  45.33	
 4   48.78  48.85	
 5   52.34  52.37	
 6   57.02  57.00	
 7   61.92  61.82	
 8   69.94  69.78	
 9   72.39  72.19	
10   79.75  79.42
```

## 문제 2
다음 두 입력 특징(비료와 살충제)을 가진 세 관측으로 구성된 시험 데이터세트의 예측을 계산하도록 5절의 코드를 수정하십시오. dataTest = torch.Tensor{ {6, 4}, {10, 5}, {14, 8} }. 세 파라미터의 값은 무엇입니까?

다음 코드를 예제 코드의 마지막 부분에 추가합니다.

```lua
print('Prediction for new data')
dataTest= torch.Tensor{{6,4},{10,5},{14,8}}

x, dl_dx = model:getParameters()
print('Model parameters of x:\n',x)
print('Model parameters of dl_dx:\n',dl_dx)

print('id  approx   text')
for i = 1,(#dataTest)[1] do
   local myPrediction = model:forward(dataTest[i])
   print(string.format("%2d  %6.2f", i, myPrediction[1]))
end
```

실행 결과:
```lua
Prediction for new data	
Model parameters of x:
  0.6660
  1.1161
 31.6428
[torch.DoubleTensor of size 3]

Model parameters of dl_dx:
-23.3542
-17.5156
 -0.7298
[torch.DoubleTensor of size 3]

id  approx   text	
 1   40.10	
 2   43.88	
 3   49.90	
```

에포크(epoch) 횟수가 1e3(1,000) 또는 1e5(100,000)일 때, 파라미터와 예측은 어떻게 달라집니까?

epoch=1e3:

```bash
id  approx   text	
 1   33.32  40.32	-- 우리의 예측과 정답 사이에 차이가 꽤 있습니다.
 2   40.65  42.92	
 3   44.08  45.33	
 4   47.28  48.85	
 5   50.47  52.37	
 6   53.44  57.00	
 7   60.30  61.82	
 8   62.56  69.78	
 9   65.99  72.19	
10   76.28  79.42	
Prediction for new data	
Model parameters of x:
  1.8319
 -0.2339
 23.2689
[torch.DoubleTensor of size 3]

Model parameters of dl_dx:
 396.5478
 297.4108
  12.3921
[torch.DoubleTensor of size 3]

id  approx   text	
 1   33.32	
 2   40.42	
 3   47.04	
```

epoch=1e5

```bash
id  approx   text	
 1   40.32  40.32	-- 우리의 예측이 정답과 거의 같아졌습니다.
 2   42.91  42.92	
 3   45.32  45.33	
 4   48.84  48.85	
 5   52.37  52.37	
 6   57.01  57.00	
 7   61.82  61.82	
 8   69.81  69.78	
 9   72.22  72.19	
10   79.44  79.42	

Prediction for new data	
Model parameters of x:
  0.6467
  1.1152
 31.9836
[torch.DoubleTensor of size 3]

Model parameters of dl_dx:
-36.7937
-27.5953
 -1.1498
[torch.DoubleTensor of size 3]

id  approx   text	
 1   40.32   -- 따라서, 새로운 데이터에 대한 예측도 더 정확하리라 추정할 수 있습니다.	
 2   44.03	
 3   49.96	
```


## 문제 3
같은 데이터세트를 사용하여 최소 제곱법의 해 θ = (X^T * X)^−1 * X^T * y를 구현하십시오. 위 시험 세트에 대한 예측은 무엇입니까? 그 예측은 SGD로 훈련된 선형 뉴런의 예측과 얼마나 다릅니까? 파라미터들은 얼마나 다릅니까?

```lua
--  {옥수수, 비료, 살충제} ->  {정답, 인자1, 인자2} -> 2차 방정식 -> y = w1*x1 + w2*x2 + w3*1(bias)
data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

X= torch.Tensor((#data)[1],(#data)[2]-1+1)  -- subtract corn col. and add bias col.
X[ { {}, 1 } ]= data[ { {}, 2 }]                   -- x1: fertilizer
X[ { {}, 2 } ]= data[ { {}, 3 }]                   -- x2: insecticide
X[ { {}, 3 } ]= torch.ones((#data)[1],1)  -- bias
y= data[ { {}, 1 }]
X_t= X:t()
-- w= torch.inverse(X_t * X) * X_t * y    -- 계산은 되지만 속도가 느림.
w= torch.mv( torch.mm( torch.inverse( torch.mm(X_t,X) ), X_t), y)  -- 복잡하지만 속도가 빠름.
th> w
  0.6501
  1.1099
 31.9807
[torch.DoubleTensor of size 3]

-- 새 데이터에 대한 예측 (weight를 dataTest의 각 열에 요소별 곱셈한 결과)
> dataTest= torch.Tensor{{6,4},{10,5},{14,8}}
> y= (dataTest[{{},1}]*w[1]) + (dataTest[{{},2}]*w[2]) + (1*w[3])
> y
 40.3204
 44.0305
 49.9603
        [torch.DoubleTensor of size 3]
```

