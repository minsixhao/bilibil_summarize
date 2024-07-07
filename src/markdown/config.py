import os
from uuid import uuid4
unique_id = uuid4().hex[0:8]

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = "tvly-"


DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
BASE_URL = '/Users/mins/Desktop/github/bilibili_summarize/static'
COOKIE_PATH = '/bilibili_summarize/cookie/cookie.json'



# {'topics_keywords': "+['鲁迅的伟大', '鲁迅在当代的地位', '鲁迅的阅读与理解', '鲁迅的孤独', '鲁迅的批判性', '鲁迅的启蒙', '鲁迅的反思', '鲁迅的自我批判', '鲁迅的遗愿']+['鲁迅', '伟大', '孤独', '批判性',性', '名言', '家喻户晓', '阅读与理解', '图腾和象征', '社会批判', '遗愿', '2024年', '社会变好']"}}
TOPICS_KEYWORDS = ['鲁迅的伟大', '鲁迅在当代的地位', '鲁迅的阅读与理解', '鲁迅的孤独', '鲁迅的批判性', '鲁迅的启蒙', '鲁迅的反思', '鲁迅的自我批判', '鲁迅的遗愿', '鲁迅', '伟大', '孤独', '批判性', '名言', '家喻户晓', '阅读与理解', '图腾和象征', '社会批判', '遗愿', '2024年', '社会变好']

TOPIC = "鲁迅的作品及思想"
CONTENT_MD = """
# 鲁迅的伟大与孤独

## 引言

咱们这节课还有一点点时间，我跟大家聊聊鲁迅。你平时在读书的时候会读鲁迅吗？课本上没有的。我年轻的时候也很不喜欢读鲁迅，一方面是不懂，另一方面是离我太远。但是很多年之后，你再回看鲁迅才能发现，我们这个时代一直在往前走，但是从来没有走出过鲁迅。

## 鲁迅的伟大

我们评价鲁迅经常用的一个词就是**伟大**。伟大的作家有一个硬性标志，就是他的伟大是伴随着读者的年纪，甚至伴随着我们时代的年纪。你在每一个年龄和每一个时代，都能从他那里获得一些新的发现，而鲁迅就是这样的作家。

## 鲁迅在当代的地位

### 热门还是冷门？

问大家两个最基本的问题：

1. 你觉得鲁迅在当代是一个热门还是冷门的作家？
    - 在高中提名的，起码在高中阶段是热门的。
    - 他当然是一个热门的作家。

### 鲁迅的知名度

鲁迅的名言很多，每个人都曾经听过：

- **愿中国的青年摆脱冷气，只是向上走。**
- **我家门前有两棵树，一棵是，另一棵还是。**
- **今天上午上了两节课，一门是数学，另一门还是数学。**

像他这样知名度很高、家喻户晓的作品，在我们中国其实是很少见的。好像你一谈到鲁迅，所有的人都能够去侃侃而谈，都能发表你的观点。

## 鲁迅的阅读与理解

读过鲁迅的人很多，但是真正读懂鲁迅的人很少。我们把鲁迅当成了一个图腾和一个象征，他被高高地挂起来了，作为一个自我标榜的标志，或者是用来炫耀自己读书读得多，点缀自己的一种谈资。很少会有人去认真的阅读体会，并且试图去理解他。

无论我们今天是冷落鲁迅，还是所谓的抬高鲁迅，我们始终都没有进入鲁迅。换一句话说，鲁迅是用他自己这种独立的人格去对抗整个民族，根植在我们的性格当中的一些劣根性。所以说这也就注定了，不论在哪一个时代，鲁迅肯定都是一个孤独的人。大众没有办法去跟他打交道，当然他也无法和大众打交道。**孤独的鲁迅只和孤独的读者打交道。**

## 鲁迅的孤独

### 鲁迅的批判性

第二个问题：鲁迅为什么孤独？你说鲁迅最明显的特点是什么？

- **批判自己**。非常好，他的文字很具有批判性，给你一种非常有力量的疼痛感。

### 批判对象

他都批判谁呢？

- **批判国民党的反动派**，批判统治阶级。
- **批判人民**，也就是被统治者。

他是一个跟其他作家唱反调的人。当大部分的作家都在界定敌人是谁的时候，鲁迅十分清醒冷静地问了一句：**我是谁？我有没有问题？**

### 鲁迅的启蒙

鲁迅不仅是我们所谓的民族魂的一个战士和斗士，他更是一个启蒙的人。他是当时那个时代黑暗中的聚火和迷蒙里的光亮。也正因为他在问**我是谁**，所以他才能够发现我们这个民族所存在的病根。

鲁迅发现我们一向是习惯于安稳于，甚至是麻木于去做一个奴隶的。尽管被压迫，离奇分歧反抗，我们更多会选择一种自我粉饰和一种自我安慰。

- **阿Q此时被人打了，但是他没法跟人家对抗，他还不了手。阿Q心里跟自己说了一句：儿子打老子了。在想象中却获得一种精神的满足和胜利，而不会去做一些失去性的改变。**

鲁迅还发现我们非常习惯于去冷眼旁观别人的痛苦，甚至是把别人的痛苦当做一种享乐。

- **我们很喜欢粉饰太平，喜欢沉默是金，喜欢安稳从众，喜欢自我感觉很良好。**

就算有什么不公，察觉到了什么问题，我们就会告诉自己：**从来如此，一直都是这样的。**

但是鲁迅偏偏要问一句：**从来如此便对吗？**

### 鲁迅的反思

鲁迅偏偏要去撕毁这层覆盖在我们生活表面的一层假面，他就要和大众唱反调，给这种摊起来太平的愉快的美满的生活增添一点烦恼，让本来都安心的愉快的多日的人有一天不愉快，揭开我们泰然自得和问心无愧背后的这种虚伪。

这就是鲁迅。当然更重要的，我觉得鲁迅身上最光辉的一点就是**他能够去反思自己**。指责别人是很简单的一件事情，但是批判自己很难很难。

- **在《狂人日记》里面，狂人提到过：中国是一个什么样的社会？人吃人的社会。他说四千年来，吃吃吃人的地方。今天知道我也在其中混了很多年，我也是吃人的一员。**

所以潜力群说：**鲁迅是一个真正可以随便谈谈的朋友，他是一个真的人，他敢于面对自己自身的局限，更无情地去解放自己。**

他的所有批判，出发点在哪里？他想改变人，而他想改变人们的原因就在于他心里真正的热爱人。所以鲁迅批判正是他爱人类的一种方式。

我们通常都觉得爱是一种很温情的状态，但是鲁迅不是。**鲁迅爱人类的方式就是批判**。他在忏悔自己的时候，揭露的是整个社会，所有人身上这个民族的病根。他要的是我们自己去反省，成为一个有自我意识和独立意识的人。

### 鲁迅的自我批判

但是问题就来了，我们都不愿意承认自己不好，都觉得自己特别好，觉得自己与众不同，觉得自己世界第一，觉得自己特别厉害。我们总是把自己看得太好了，看得太高了。所以我们想象不到一个人可以像鲁迅一样，他去挖苦自己，他用一把刀去挖出自己血泪挖出来给别人看。

我们理解不了，居然会有这样的人，所以我们只能把他的这种血泪和忏悔解释成他在恶意伤人。所以我们去责备他，太刻薄了，太不厚道了。

但是鲁迅尽管面临着当时所有人对于他的指责，对于他的不理解，他依然选择了去做一个真的猛士。

- **鲁迅在《慕杰伦》里面写过一句我特别特别喜欢的话：于浩哥狂热之际中寒，于天上看见深渊，于一切眼中看见无所有，于无所希望中得救。**

他要从我们粉饰出来的这种看起来很美好的世界里，去看见这个世界所存在的黑暗的深渊和我们身上所散发出的寒气。

所以于广安提到说：**这就是我为什么热爱鲁迅的理由。他的叙述在抵达现实时是如此的迅猛，就像子弹穿越了身体而不是留在身体里。**

我们可以说鲁迅的子弹就是非常精确的，而且毫无犹豫地去洞穿我们的身体。他给我们留下了一个洞，从肩膀后一直有寒风呼啸地从我们身体上这个流着血的伤口穿过，并且这个伤口永远都不会愈合。

但是鲁迅的文字，他留下的这个伤口，它能够让我们时刻去保持清醒，保持疼痛，让我们能从中去看到自己的魂灵。

## 鲁迅的遗愿

我们中国的士大夫有一个特点就是，我们特别希望名垂青史，希望整个后世记住我们的名字。但是鲁迅是不一样的，他说：**不要做任何关于纪念的事情**。他觉得自己的书被后人高度评价不是一件好事，这只能说明国家和民族仍然没有悔改。他盼望自己的书真正被人遗忘的那一天早点到来。

## 结语

而我们看现在2024年，一百多年过去了，当我们再一次翻开鲁迅的时候，你会发现我们还是他笔下的那些人物，我们还生活在他笔下的社会当中。这个社会还有那么多的旁观者离乡了，那么多的人在嘲讽在奚落，在把对人的痛苦当成自己的快乐。

所以我也在想，这个社会真正变好了吗？我也不知道这个问题。但是我们还是向往着，并且期待着变好的那一天的到来。

就好像鲁迅一直被人所喜爱的一句话：**此后如今没有去火，我们也是唯一的光。**
"""
OLD_OUTLINE = """

# 鲁迅的伟大与孤独

## 引言

鲁迅的作品在年轻时不易理解，但随着时间推移，我们发现时代从未走出过鲁迅的影响。

## 鲁迅的伟大

鲁迅的伟大在于他的作品能在不同年龄和时代带来新的发现。

## 鲁迅在当代的地位

### 热门还是冷门？

鲁迅在高中阶段是热门作家。

### 鲁迅的知名度

鲁迅的名言广为人知，家喻户晓，大家都能谈论他。

## 鲁迅的阅读与理解

读过鲁迅的人多，但真正理解他的人少。鲁迅被当作图腾和象征，很少有人认真阅读和理解他。鲁迅用独立人格对抗民族劣根性，注定孤独。

## 鲁迅的孤独

### 鲁迅的批判性

鲁迅的文字具有批判性，带来疼痛感。

### 批判对象

鲁迅批判国民党反动派、统治阶级和被统治者。他质问“我是谁？我有没有问题？”

### 鲁迅的启蒙

鲁迅是启蒙者，揭示民族病根，质问“从来如此便对吗？”

### 鲁迅的反思

鲁迅撕毁生活表面的假面，反思自己，批判社会。他的批判源于对人类的热爱。

### 鲁迅的自我批判

鲁迅敢于自我批判，揭露社会病根。他的文字让人保持清醒和疼痛。

## 鲁迅的遗愿

鲁迅希望不要纪念他，书被高度评价说明国家和民族没有悔改。

## 结语

2024年，我们仍生活在鲁迅笔下的社会中，期待社会变好。鲁迅的话：“此后如今没有去火，我们也是唯一的光。”

"""


CONVERSATION = {
    "conversations": "### Subject_Matter_Expert\n\n听说你在写一篇关于 鲁迅的作品及思想 的文章？\n\n### Xu\n\n是的，我正在研究鲁迅的作品和思想。作为一名专注于现代中国文学的教授，我对鲁迅的作品有着浓厚的兴趣。为了更深入地理解他的作品，我希望能从你这里获取一些专业的见解。\n\n首先，我想问一下，在鲁迅的作品中，您认为最能体现他对社会批判和改革思想的作品是哪一部？为什么？\n\n### Subject_Matter_Expert\n\n在鲁迅的作品中，《狂人日记》被广泛认为是最能体现他对社会批判和改革思想的一部作品。这部小说是中国现代文学史上第一篇白话文小说，以一种独特的视角揭示了封建礼教对人性的压迫和扭曲。通过狂人的眼睛，鲁迅揭露了社会中的虚伪、冷漠和残酷，强调了个体在封建制度下的痛苦与无奈。这部作品通过狂人对“人吃人”社会的控诉，传达了鲁迅对封建社会的深刻批判和对社会改革的强烈渴望。\n\nCitations:\n\n[1]: https://baijiahao.baidu.com/s?id=1801265659583320272\n[2]: https://www.jianshu.com/p/e5462c2becc7\n\n### Xu\n\n谢谢你的解答。《狂人日记》确实是一部重要的作品，揭示了许多深刻的社会问题。接下来，我想了解一下鲁迅在《阿Q正传》中通过阿Q这个角色，具体如何表现出对中国社会和国民性的批判？\n\n### Subject_Matter_Expert\n\n在《阿Q正传》中，鲁迅通过塑造阿Q这个角色，深刻揭示了中国社会的种种弊病和国民性的问题。阿Q是一个贫苦的农民，他的生活充满了屈辱和压迫，但他却常常用“精神胜利法”来逃避现实。这种自欺欺人、安于现状的态度，正是鲁迅对中国社会和国民性的一种批判。\n\n首先，阿Q的“精神胜利法”是对中国人自我安慰、自我麻痹的讽刺。阿Q每次受到挫折或羞辱后，总是通过自我安慰来恢复心理平衡，这种行为反映了社会中普遍存在的麻木和不愿直面现实的心态。\n\n其次，阿Q的行为和命运揭示了社会底层人民的无奈和悲哀。他的贫穷和无知使得他无法改变自己的命运，而社会的冷漠和压迫更是让他在绝望中挣扎。这反映了鲁迅对社会不公和人性冷漠的批判。\n\n最后，通过阿Q的悲剧命运，鲁迅表达了对社会改革的迫切希望。阿Q最终在革命中被误杀，这一情节不仅揭示了革命的不彻底性和盲目性，同时也表达了鲁迅对社会变革的深刻思考。\n\n总的来说，《阿Q正传》通过阿Q的形象，揭示了中国社会的虚伪、冷漠和残酷，批判了国民性中的自欺、麻木和懦弱，表达了鲁迅对社会改革的强烈渴望。\n\nCitations:\n[1]: https://www.sohu.com/a/717510010_121778433\n\n### Xu\n\n这些见解非常有帮助，谢谢你！接下来，我想更深入地探讨鲁迅对传统文化的态度。在他的作品中，他如何表现出对传统文化的批判？有没有具体的例子可以说明？\n\n### Subject_Matter_Expert\n\n鲁迅的作品中，通过多个角色和情节，对传统文化进行了深刻的批判。其中最具代表性的作品包括《狂人日记》、《阿Q正传》和《高老夫子》。\n\n在《狂人日记》中，鲁迅通过一个精神失常的“狂人”的视角，揭示了封建礼教的吃人本质。狂人在日记中不断提到“吃人”，这不仅是对封建礼教的控诉，也是对整个封建社会的批判[1]。\n\n《阿Q正传》通过阿Q这个底层农民的形象，批判了中国传统社会中自欺欺人的国民性。阿Q的“精神胜利法”正是对传统文化中逃避现实、自我安慰的态度的讽刺[2]。\n\n《高老夫子》通过描写一个伪道学家高尔础的荒诞经历，揭示了封建文化和教育的虚伪。高尔础追求名利，放弃尊严，最终一事无成，这不仅是对个人的批判，更是对封建文化和教育制度的深刻讽刺[3]。\n\n综上所述，鲁迅在他的作品中，通过不同的角色和情节，揭示了传统文化的种种弊病，表达了对封建礼教、虚伪道学和国民性的深刻批判。\n\nCitations:\n\n[1]: https://baijiahao.baidu.com/s?id=1801265659583320272\n\n[2]: https://www.sohu.com/a/717510010_121778433\n\n[3]: https://baijiahao.baidu.com/s?id=1798715330997096249\n\n### Xu\n\n这些例子确实很有说服力，帮助我更好地理解了鲁迅对传统文化的批判。接下来，我想了解一下鲁迅在作品中如何运用象征主义和隐喻来传达他的思想？能否提供一些具体的例子来说明这一点？\n\n### Subject_Matter_Expert\n\n鲁迅在作品中运用象征主义和隐喻来传达他的思想，丰富了作品的层次和内涵。《狂人日记》和《阿Q正传》是其中的两个典型例子。\n\n在《狂人日记》中，鲁迅通过“月光”这一意象，表达了对封建礼教的控诉。月光在作品中多次出现，如“今天晚上，很好的月光”，象征着表面上的宁静和美好，但实际上揭示了社会的黑暗和残酷[1]。\n\n在《阿Q正传》中，鲁迅通过阿Q的“辫子”这一意象，反映了封建社会对个体的束缚和压迫。阿Q的辫子不仅是他身份的象征，也是他屈从于封建礼教的表现。这一细节揭示了社会对个体的控制和压迫[2]。\n\n此外，在《肥皂》中，鲁迅巧妙地运用了“肥皂”这一象征，揭示了社会转型期人们复杂的心理状态和内心矛盾。肥皂象征着性和欲望，通过这一意象，鲁迅展示了人们在传统与现代之间的挣扎与冲突[3]。\n\n总的来说，鲁迅通过运用象征主义和隐喻，不仅增强了作品的艺术性，也深刻揭示了社会问题和人性的复杂性。\n\nCitations:\n\n[1]: https://www.lingceed.com/blog/ib-chinese-luxun\n[2]: https://baijiahao.baidu.com/s?id=1801265659583320272\n[3]: http://cll.newdu.com/m/view.php?aid=28262\n"
}


OUTLINE = """
# 鲁迅的作品及思想

## 鲁迅：中国现代文学的奠基人
概述鲁迅作为中国现代文学奠基人的地位及其重要性。

## 鲁迅的生平与思想背景
介绍鲁迅的生平及其思想形成的背景。

### 早年生活与教育经历
鲁迅的家庭背景、早年生活和教育经历。

### 日本留学时期的思想转变
鲁迅在日本留学期间的思想变化及其对后来的影响。

### 对中国传统文化的批判与反思
鲁迅对中国传统文化的批判和反思。

## 鲁迅的文学创作
鲁迅在小说、杂文、散文及诗歌等方面的创作。

### 小说作品
鲁迅的主要小说及其分析。

### 杂文与散文
鲁迅的杂文和散文创作。

### 诗歌与其他
鲁迅在诗歌创作和翻译工作方面的贡献。

## 鲁迅的思想与影响
鲁迅的文学思想、文化观、社会观及其教育思想。

### 文学思想
鲁迅的现实主义与批判精神，对封建主义与迷信的批判。

### 文化观
鲁迅对传统文化的继承与创新，以及对现代文化的贡献与期望。

### 社会观
鲁迅对社会不公与阶级矛盾的揭示，对民族解放与社会进步的思考。

### 教育思想
鲁迅对教育改革的主张，对青年的期望与指导。

## 鲁迅作品的现代意义
鲁迅作品的文学价值、时代意义及其国际影响。

### 文学价值
鲁迅作品在中国现代文学史上的地位及其艺术特色与创新。

### 时代意义
鲁迅思想对当代社会的启示及其在现代文化中的传承与影响。

### 国际影响
鲁迅作品的海外传播与接受及其在世界文学中的地位。

## 结语
总结鲁迅作品与思想的重要性及其在中国乃至世界文化中的贡献。
"""