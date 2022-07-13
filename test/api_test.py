import requests

API_URL = "https://kospacing.nextlab.ai/kospacing/predict"

params = {
        "sentences": [
            "그것도 그렇지만 한달간 유럽 다니면서 3번 추락했던 에어2s와 다르게 반년 가까이 동남아를 여행하고 있지만 단 한번도 추락을 한 적도 없는 안정성도 비교가 되지 않을 것 같기도 합니다. 일례로 커다란 비행기가 작은 비행기보다 더 안전하다는 말도 있죠. 특별한 설명이 없어도 이해하시리라 생각합니다. 연결거리도 상당히 차이가 나는 것 같구요."
        ]
        # "sentences": [
        #     "일본은임진왜란을일으켰으나이순신장군에의해실패했다.",
        #     "윤석열은김건희때문에대한민국역사상두번째탄핵대통령이될가능성이높아졌다."
        # ]
    }

response = requests.post(API_URL, json=params)

result_sentences = response.json()

print(result_sentences)
# ['일본은 임진왜란을 일으켰으나 이순신 장군에 의해 실패했다.', '윤석열은 김건희 때문에 대한민국 역사상 두 번째 탄핵 대통령이 될 가능성이 높아졌다.']