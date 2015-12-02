# 실습 1
기계 학습, 2015년 봄

# 설치
- [토치(Torch) 시작하기](https://docs.google.com/document/d/18sTMqVFDSFvNaz8xIo40Wm9XBSrMX8spTDGr4F0XXi8/edit#/)
- [루아 15분 안에 배우기](http://roboticist.tistory.com/576)
- [텐서](https://docs.google.com/document/d/1QDtM8pdduWeUw0UK4zd7KgTuC6mFEbFCfVd3TF7Vj_Q/edit)
- [수학 연산들](https://docs.google.com/document/d/1js0VjoZ4HzixMVQvx7Lxw75ORCi-kRFoNqNrqFcUuNQ/edit) 

## 문제 1
다음 텐서 t에서 두 번째 열에 있는 요소들을 추출하는 세 가지 방식을 나열하시오.

```lua
th> t=torch.Tensor{{1,2,3},{4,5,6},{7,8,9}}
th> t
 1  2  3
 4  5  6
 7  8  9
[torch.DoubleTensor of size 3x3]

-- 방법 1
th> t:narrow(2,2,1)
 2
 5
 8
[torch.DoubleTensor of size 3x1]

-- 방법 2
th> t:sub(1,3,2,2)
 2
 5
 8
[torch.DoubleTensor of size 3x1]

-- 방법 3
th> t:select(2,2)
 2
 5
 8
[torch.DoubleTensor of size 3]

-- 방법 4
th> t:index(2,torch.LongTensor{2})
 2
 5
 8
[torch.DoubleTensor of size 3x1]

-- 방법 5
th> t[{{},2}]
 2
 5
 8
[torch.DoubleTensor of size 3] 
```

서브텐서 추출에 대해서는 [이 문서](https://docs.google.com/document/d/1QDtM8pdduWeUw0UK4zd7KgTuC6mFEbFCfVd3TF7Vj_Q/edit#heading=h.6w)를 참고하십시오.


## 문제 2 
텐서와 스토리지의 차이: 스토리지는 c 언어의 배열 같은 메모리와 밀접하게 관련된 자료 구조입니다. 반면, 텐서는 그 스토리지를 바라보는 한 방법입니다. 