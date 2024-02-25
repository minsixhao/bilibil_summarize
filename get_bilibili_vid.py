import httpx
import json

def load_cookie() -> dict:
    """用于加载cookie"""
    try:
        file = open('cookie/cookie.json', 'r')
        cookie = dict(json.load(file))
    except FileNotFoundError:
        msg = '未查询到 Cookie 文件，请确认资源完整'
        cookie = {}
        print(msg)
    return cookie


def get_bvids():
    url = 'https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/all?type=video'
    cookie = load_cookie()
    with httpx.Client() as client:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
        }
        data = client.get(url=url, headers=headers, cookies=cookie)
        data = data.json()

    bvids = []
    for item in data["data"]["items"]:
        dynamic_module = item["modules"]["module_dynamic"]["major"]

        if dynamic_module is None or "archive" not in dynamic_module or dynamic_module["archive"] is None:
            continue

        bvid_value = dynamic_module["archive"]["bvid"]
        bvids.append(bvid_value)
    return bvids



