import requests
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class JinaAI():
    def __init__(self):
        self.apiKey = 'jina_3cb0edd3307c4d9d9893447cb95eff92zjurUstt7_JVCjY0nql9lgGap2Jl'
        self.readerBaseUrl = 'https://r.jina.ai/'
        self.searchBaseUrl = 'https://s.jina.ai/'

    def reader(self, url: str):
        readerUrl = self.readerBaseUrl + url
        try:
            response = requests.get(readerUrl, headers={
                "Authorization": f"Bearer {self.apiKey}",
                # "X-Return-Format": "text"
            })
            response.raise_for_status()  # Ensure we raise an error for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Exception for URL {readerUrl}: {e}")
            return None

    def search(self, url: str):
        searchUrl = self.searchBaseUrl + url
        try:
            response = requests.get(searchUrl, headers={"Authorization": f"Bearer {self.apiKey}"})
            response.raise_for_status()  # Ensure we raise an error for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Exception for URL {searchUrl}: {e}")
            return None
    
    def url_summary(self, url: str):
        loader = SeleniumURLLoader(urls=[url])
        try:
            data = loader.load()
            return data[0]
        except Exception as e:
            logging.error(f"Exception for URL {url}: {e}")
            return None
    


# reader_result = JinaAI().reader('https://m.gushiwen.cn/gushiwen_0fd00ff1aa.aspx')
# print(reader_result)
#
# search_result = JinaAI().search('https://www.sohu.com/a/656087478_121123846')
# print(search_result)

# search_result = JinaAI().url_summary('https://www.sohu.com/a/656087478_121123846')
# print(search_result)

# from config import MILVUSLOADPTAH
# from url_read_search import JinaAI
# from milvus_load_retrieval import MilvusLoadRetrieval

# sourceReader = JinaAI()

# sources = ['https://www.eduxiao.com/riji/111422.html']
# for source in sources:
#     # content = sourceReader.reader(source)
#     content = """
# 鲁迅，小名阿张，本名樟寿，初字豫山，后字豫才，改名周树人[1]。1918年（民国七年）发表第一篇白话小说《狂人日记》时候开始使用“鲁迅”作为笔名。1881年9月25日（光绪七年八月初三），出生在浙江省绍兴府会稽县（今绍兴）。著名文学家、思想家、革命家、教育家、民主战士，新文化运动的重要参与者，中国现代文学的奠基人之一。[2]
# 早年赴日本学医，后弃医从文并参加光复会。辛亥革命后，曾任南京临时政府和北京政府教育部部员等职。五四运动前后，提倡新文化，反对旧文化，参加《新青年》编辑部工作。1920年（民国九年），在北京大学、北京女子师范大学军校兼课。此后相继出版了《呐喊》、《彷徨》、《阿Q正传》等。1926年（民国十五年）后，曾在厦门大学、中山大学等校任教。1927年（民国十六年）10月之后，定居上海，研究翻译马克思主义文艺理论。1930年（民国十九年）起，先后参加中国自由运动大同盟、中国左翼作家联盟，中国民权保障同盟等组织，成为左翼文学的领袖人物。1936年（民国二十五年）10月19日病逝于上海。[3]
# 鲁迅一生在文学创作、文学批评、思想研究、文学史研究、翻译、美术理论引进、基础科学介绍和古籍校勘与研究等多个领域具有重大贡献，并在著文翻译、辑校文献、文稿校改、日记书信、诗稿题赠、设计装帧等方面留下了非常丰富的墨迹。他对于五四运动以后的中国社会思想文化发展具有重大影响，蜚声世界文坛，尤其在韩国、日本思想文化领域有极其重要的地位和影响，被誉为“二十世纪东亚文化地图上占最大领土的作家”。[4]
# 人物生平
# 少年时期
# 鲁迅，1881年9月25日（光绪七年八月初三），出生在浙江省绍兴府会稽县（今绍兴）一个传统士大夫家庭，祖父周介孚在京中做官，父亲周伯宜是秀才。鲁迅七岁进本宅私塾就读，十二岁转入绍兴全城最严格的书塾“三味书屋”。少年鲁迅在接受严格的传统文化教育的同时，还喜欢涉猎野史、笔记、神话小说之类的书籍。1893年（光绪十九年），鲁迅十三岁时，祖父因科场作弊案下狱。官府趁机敲诈勒索，父亲身患重病，鲁迅的家境由小康坠入困顿。在家庭破落的过程中所显露出来的人情冷暖、世态炎凉，使少年鲁迅开始体会到上层社会的虚伪和腐败。在此期间，鲁迅一度寄居在城郊的外婆家。在那里，他结识了很多农民朋友。这一段经历为他日后创作以农民生活为题材的作品奠定了生活基础。[5]
# image
# 少年时期鲁迅
# 求学之路

# 鲁迅的求学之路
# 1898年5月（光绪二十四年），鲁迅到南京去投考无需学费的学校，进入江南水师学堂，被编入管轮班。11月，鲁迅离开水师学堂，回乡省亲。年底参加会稽县试。但鲁迅对“博取功名”不感兴趣，没有参加府试。次年1月重返南京，改入江南陆师学堂附设的矿务铁路学堂。1902年1月（光绪二十八年），以一等第三名优异成绩毕业。在南京求学时期，鲁迅接触了西方近代思潮。他钟爱《时务报》《译学汇编》，卢梭、孟德斯鸠、斯宾塞的著作，林琴南翻译的外国小说。《天演论》中“物竞天择”、“优胜劣败”等关于发展变化的观点，初步形成了他早期进化论的社会发展观。[6]
# 在矿务铁路学堂毕业后，鲁迅被两江总督派赴日本留学。1902年4月（光绪二十八年）抵日本东京，入弘文学院普通科江南班。在这里，他结识了许寿裳，此后两人建立了深厚的友谊。作为当时反清阵营的坚定支持者，鲁迅带头剪了辫子，经常参加反清革命民主派的各种集会。这期间他编译了历史小说《斯巴达之魂》，编写了论述中国地质和矿产分布情况的专著《中国矿产志》，发表了科学论文《中国地质略论》《说鈿》，翻译出版了科学幻想小说《月界旅行》、《地底旅行》。1904年4月（光绪三十年），鲁迅从弘文学院速成普通科毕业，同年9月入仙台医学专门学校。后因在课堂上观看日俄战争影片，感受到了身为弱国国民的悲愤，从此改变了医学救国的思想，转而志向于文学，以拯救中华民族的灵魂为急务。于1906年3月（光绪三十二年），重返东京，学籍列在东京德语学校，从事文学事业。[7]
# 1906年6月（光绪三十二年），鲁迅结束了在仙台医专的学业，自日本回绍兴，奉母命与朱安女士结婚。在绍兴住了几天之后回到东京，一面继续学习外文，一面从事文学活动。1907年（光绪三十三年），与许寿裳计划筹办文艺杂志《新生》，没有成功。随后，他在刘师培等人主编的《河南》杂志上发表了《人间之历史》《文化偏至论》《摩罗诗力说》等论文，标志他独立思想的逐步形成。1909年（光绪三十四年），鲁迅和周作人合译《域外小说集》，介绍俄国和东欧国家的一些短篇小说，是他译介外国进步文学作品的开端。这期间，鲁迅同革命党人陶成章等人时有过从，参加光复会为会员，后师从章太炎学文字学。[8]
# image
# 鲁迅（后排左）在日本留学
# 四处任教
# 1909年8月（宣统元年），鲁迅因家境困难，回国谋职。先在杭州的浙江两级师范学堂任生理学和化学教员，1910年7月又到绍兴府中学堂任生物学教员并兼任监学，课余辑录类书中唐以前的小说，后定名为《古小说钩沉》。辛亥革命爆发时，他曾积极组织声援活动和宣传活动。1911年10月，任绍兴山会初级师范学堂校长。这年冬天，他以辛亥革命为背景创作了他的第一篇小说《怀旧》。[9]
# 中华民国临时政府成立之后，1912年2月（民国元年），鲁迅应教育总长蔡元培之邀到南京教育部工作，不久随部迁至北京，担任教育部佥[qiān]事、社会教育司第二科科长、第一科科长等职，主管文化及社会文化设施等工作。公余时间，鲁迅还辑录唐宋短篇小说，后辑成《唐宋传奇集》。辛亥革命之后，中国的社会状况并没有发生根本变化，袁世凯称帝、张勋复辟等倒行逆施充分暴露了中国旧文化的顽固性，鲁迅一时找不到攻击旧文化的机会和战友，常常感到极度的苦闷和绝望，一度倾心于辑录、校勘古籍，搜集金石碑帖，研究佛经。在担任教育部社会教育司第一科科长兼教育部佥事期间，鲁迅在提倡美育、制定注音字母方案、开展通俗教育、筹创京师图书馆和历史博物馆等方面，都做了许多有益的工作。[10]
# 1918年（民国六年）初，鲁迅参加《新青年》的编辑工作，结识了李大钊、陈独秀和胡适等人，投身于五四新文化运动。[11]
# 文坛先声

# 新青年
# 1918年（民国六年）5月，鲁迅在《新青年》上发表了中国现代文学史上第一篇现代白话小说《狂人日记》，揭开了中国小说史上新的一页。五四期间，鲁迅又陆续发表了《孔乙己》《药》《明天》等多篇小说。1922年（民国十一年）初，持文化保守立场的“学衡派”，从学理上反对新文化运动，发表对“新文化”走向的不同见地，引发了围绕中国新文化问题的论争。鲁迅撰写《估<学衡>》等杂文，回应“学衡派”众人，驳斥《学衡》的理论主张。这些杂文大都收在杂文集《热风》和《坟》里。[12]
# image
# 《新青年》
# 从1920年（民国九年）秋季开始，鲁迅在北京大学、兼任北京女子师范学校、世界语学校教师，讲授中国小说史等课程。后来他把讲义整理《中国小说史略》于公开出版，这是第一部比较系统地论述我国小说发展历史的专著。1923年（民国十二年），鲁迅的第一部短篇小说集《呐喊》出版。1924年（民国十三年），鲁迅还应邀到陕西西安大学讲授“中国小说的历史的变迁”。从1920年至1926年（民国九年至民国十五年），鲁迅先后在北京八所大中学校兼课。[13]
# 为了培育文艺新苗，广泛制造“批评社会，批评文明”的进步舆论。1924年（民国十二年）底，鲁迅参与了《语丝》周刊的创办，并参加了语丝社。1925年（民国十四年）又先后组织和领导了莽原社和未名社，帮助支持了《晨报副刊》《民众文艺周刊》的运作。这些社团在新文化的建设和当时的政治斗争中，都起到了重要的积极作用。与此同时，以“五卅[sà]”反帝爱国运动为标志，革命运动在1925年（民国十四年）至1926年（民国十五年）迅猛发展。鲁迅三次为“五卅”惨案捐款，参加了北京女师大学生运动和“三一八”爱国运动。[14]
# 民主战士
# 三一八惨案之后，鲁迅受到北洋政府的通缉。为了避开迫害，也出于对南方革命斗争的向往以及个人生活方面的原因，鲁迅于1926年（民国十五年）8月南下任厦门大学文科国文系教授、国学研究院研究教授，开设“小说选及小说史”、“文学史纲要”等课程。文学史讲稿后来整理成《汉文学史纲要》公开出版。在厦门大学期间，他继续写了五篇《旧事重提》，两篇《故事新编》，一本《两地书》以及《华盖集续编的续编》等，共十七万余字。此外，还鼓励和指导厦门大学学生组织文学团体和出版刊物。同年底，鲁迅辞去厦门大学的职务。[15]
# 1927年（民国十六年）1月18日，鲁迅抵达当时的革命中心广州，就任中山大学文学系主任兼教务主任，开设文艺论、中国小说史、中国文学史等课程，后又被特聘为中山大学组织委员会委员。同年4月12日，蒋介石在上海叛变革命，发动“四一二”政变，大肆屠杀共产党人和革命群众。4月15日，广州的国民党反动派也开始了反革命的大杀戮。鲁迅召集中山大学各主任开紧急会议，跟学校当局展开斗争，力主营救被捕学生，但遭到拒绝。不久，鲁迅愤然辞去中山大学的一切职务，继续在广州从事创作和翻译工作。10月，鲁迅离开广州赴上海，与许广平一起定居，结为终身伴侣。[16]
# 鲁迅到上海定居不久就参加了中国革命互济会，跟中国共产党取得了联系。1928年（民国十七年），创造社和太阳社发动无产阶级革命文学运动，认为“五四”以来那些重在描写与揭示生活现实的作品都已经落伍过，要彻底抛弃，新文学队伍也要按阶级属性重新划线站队。由此，他们便向“五四”时期已成名的作家开刀，全盘否定“五四”新文学的传统，认为鲁迅写作的那个“阿Q时代早已死去”，鲁迅的创作大都没有现代意味，只能代表清末及庚子义和团时代的思想，甚至判定鲁迅是“封建余孽”“二重反革命人物”。由此鲁迅跟太阳社、创造社展开了一场关于“革命文学”问题的论争。鲁迅并非反对“革命文学”，他对革命文学其实没有明确的设想，他只是怀疑和反感革命文学家的“突变”及唯我独革。鲁迅从现实的角度肯定了“革命文学”作为一种反抗性思潮的存在理由，认为这是“势所必至，平平常常，空嚷力禁，两皆无用”，同时也批评了创造社、太阳社不敢正视残酷的现实，光凭纸上写下的“打打”“杀杀”，只不过是“空嚷”而已。鲁迅对创造社诸人片面宣扬文学工具论表示反感，特别不赞同所谓“组织生活论”“工具论”，认为文艺“不过是一种社会现象，是时代人生的记录”，“现在的文艺，就在写我们自己的社会”，如果将文艺等同于政治，那就“踏着‘文学是宣传'的梯子而爬进唯心的城堡里去了”。在论争的过程中，鲁迅翻译并钻研了马克思主义文艺理论。1929年（民国十八年），鲁迅主编《科学的艺术论丛书》，先后翻译出版了普列汉诺夫的《艺术论》、卢那察尔斯基的《文艺与批评》等论著。[17]
# 1929（民国十八年）年9月，儿子周海婴在上海出生。[18]
# 左联领袖
# 革命文学的论证经历了近两年时间，引起了国共两党的注意。1929年9月，国民党召开“全国宣传会议”，提出以“三民主义的文艺政策”来清理统一文坛，扼杀“革命文学”“无产阶级文学”。共产党则指示创造社、太阳社停止攻击鲁迅，他们与鲁迅以及其他革命的“同路人”联合起来，建立统一的革命文学组织，对抗国民党的文化围剿。这样，历时近两年的论争便停止了。参与论争的各方冷静下来，寻求共识，组成了“中国左翼作家联盟”。1930年（民国十九年）3月，中国左翼作家联盟（简称“左联”）成立，会上选举了包括鲁迅在内的7人为常务委员。鉴于鲁迅当时在文学界的影响作用，曾被攻击为落后的“人道主义者”的鲁迅，此时被左联尊为左翼文学的“领袖”。鲁迅虽然被内定为左联的“盟主”，但在加入左联以后，鲁迅并没有按例参与左联的常规性的政治活动(如开会、上街游行、飞行集会等)，他首先做的是清理与创造社、太阳社论战的“旧战场”，这项工作的中心任务是翻译、介绍马克思主义文艺理论和苏联文艺政策的相关著述。在“左联”成立大会上，鲁迅做了题为《对于左冀作家联盟的意见》的讲话，清醒地总结了革命文学倡导过程中的经验教训。他针对某些革命作家盲目乐观的心态，批评那种“不明白革命的实际情形”，“不明白革命是痛苦，其中也必然混有污秽和'血'的'浪漫'"，要正视现实，摒弃浪漫蒂克的幻想。[19]
# 除了“左联”之外，鲁迅还投入各种社会活动，先后加入中国共产党发起的革命互济会、中国自由运动大同盟、中国民权保障同盟和反帝反战同盟；对国民党的压迫，帝国主义的暴行，多次和进步文化界一起发表宣言，提出抗议。1931年（民国二十年）2月，柔石、殷夫等五位青年作家被秘密杀害，传闻将搜捕鲁迅。鲁迅被迫离开寓所去别处暂避。不久，他冲破国民党当局的严密封锁，在“左联”的秘密刊物上发表文章纪念战死者，并撰文在国外报刊上揭露黑暗中国的文艺界现状。这年9月，发生“九一八”事变，中华民族处于危亡时刻。鲁迅撰写了一系列犀利的杂文，这两年间的三十七篇杂文和一篇译文，收集在《二心集》中。[20]
# 1932年（民国二十一年）初，上海爆发了“一二八”抗战。鲁迅和茅盾等四十余人联名发表《上海文艺界告世界书》，抗议日本帝国主义的暴行。1931年至1933年（民国二十年至民国二十三年）期间，鲁迅与瞿秋白从通信、见面到结为知己。瞿曾三次在鲁迅家暂住，以避敌人的追踪。得到鲁迅在工作上和生活上的许多关照。1933年（民国二十二年）1月，鲁迅担任了中国民权保障同盟上海分会的执行委员。与宋庆龄等人赴德国驻上海领事馆，递交反对希特勒法西斯暴行的抗议书。9月，世界反对帝国主义战争委员会在上海秘密召开远东反战会议，鲁迅被推为名誉主席团成员。同年，鲁迅会见了英国著名作家萧伯纳和美国进步记者斯诺。在同形形色色的敌人和思潮作斗争的过程中，鲁迅用各种笔名写作了大量战斗的杂文。这期间，鲁迅着重批判了主张“文艺自由”论的“自由人”胡秋原和自称“第三种人”的苏汶（杜衡）。[21]
# 1934年（民国二十三年）鲁迅坚持中国语文的改革和文艺大众化的方向，作《门外文谈》。同时，鲁迅作《答国际文学社问》，先后发表于国际革命作家联盟的机关刊物《国际文学》和苏联《真理报》，他把这一年写下的杂文，编为《花边文学》和《且介亭杂文》。[22]
# image
# 鲁迅
# 1935年（民国二十四年），鲁迅密切关注国内政治形势的发展动向，积极培养左翼青年作家，为叶紫、萧军、萧红的作品写序，这一年的杂文结成《且介亭杂文二集》。在这一年里，鲁迅还写了《理水》《采薇》《出关》《生死》，与1934年（民国二十三年）写的《非攻》和1927年（民国十六年）以前写的《补天》《奔月》《铸剑》一起结集为《故事新编》，于1936年（民国二十五年）出版。[23]
# 1935年（民国二十四年），鲁迅原有的肺病日渐严重，但他不愿离开战斗岗位移地疗养。1936年（民国二十五年），他虽在病中，依然勤奋工作，写了不少文章。当民族危机日益严重，中国文艺界的抗日民族统一战线急需建立时，鲁迅发表了《答徐懋[mào]庸并关于抗日统一战线问题》等文章，表示坚决拥护中共中央关于建立抗日民族统一战线的方针，提出了“民族革命战争的大众文学”的口号，与周扬等人捍卫的“国防文学"口号，展开了“两个口号”之争。[24]
# 人物逝世
# 1936年（民国二十五年）10月19日，鲁迅与世长辞。蔡元培、马相伯、宋庆龄、毛泽东、内山完造、史沫特莱、沈钧儒、矛盾、萧三组成治丧委员会。上海各界人民纷纷赴万国殡仪馆瞻仰鲁迅遗容。22日，二万余人送殡。鲁迅遗体安葬于虹桥万国公墓。1956年，鲁迅墓迁移重建于上海虹口公园。[25]
# """
#     with open(MILVUSLOADPTAH, 'w', encoding='utf-8') as file:
#         file.write(content)
#     m = MilvusLoadRetrieval()
#     m.load()
#     retrieve_content = m.retrieval('早年生活')
#     print('--')
#     print("retrieve_content:", retrieve_content)