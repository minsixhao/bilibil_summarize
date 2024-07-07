from typing import List, Dict, Union, TypedDict

# 定义句子结构
class Sentence(TypedDict):
    sentence: str
    refs: List[str]

# 定义内容结构
class ContentSection(TypedDict):
    section_title: str
    section_content: List[Sentence]
    subsections: List['ContentSection']

# 定义References结构
class References(TypedDict):
    key: str

# 定义SpicyEntities结构
class SpicyEntities(TypedDict):
    entity: str

# 定义整个结构
class Article(TypedDict):
    title: str
    url: str
    summary: str
    content: List[ContentSection]
    references: Dict[str, References]
    spicy_entities: Dict[str, SpicyEntities]
    flair_entities: List[str]

# 示例实例
example: Article = {
    "title": "2022_AFL_Grand_Final",
    "url": "https://en.wikipedia.org/wiki/2022_AFL_Grand_Final",
    "summary": "The 2022 AFL Grand Final was an Australian rules football match contested between Geelong and the Sydney Swans at the Melbourne Cricket Ground on 24 September 2022. It was the 127th grand final of the Australian Football League (AFL), staged to determine the premiers of the 2022 AFL season. The match, attended by 100,024 spectators, was won by Geelong by a margin of 81 points, marking the club's tenth VFL/AFL premiership. Isaac Smith of Geelong won the Norm Smith Medal as the player judged best on ground.",
    "content": [
        {
            "section_title": "Background",
            "section_content": [
                {
                    "sentence": "Geelong entered their 2022 campaign after a heavy defeat in the 2021 preliminary finals against eventual premiers Melbourne",
                    "refs": []
                },
                {
                    "sentence": "The Cats were inconsistent early in the season, opening with five wins and four losses, but did not lose again for the remainder of the home-and-away season, earning their second minor premiership in four seasons with 18 wins and four losses",
                    "refs": []
                },
                {
                    "sentence": "Geelong then defeated Collingwood by six points in their qualifying final and thrashed the Brisbane Lions by 71 points in the first preliminary final",
                    "refs": []
                },
                {
                    "sentence": "It was Geelong's second grand final appearance in three years",
                    "refs": []
                },
                {
                    "sentence": "Sydney had been eliminated in 2021 after a one-point defeat against local rivals Greater Western Sydney in their elimination final",
                    "refs": []
                },
                {
                    "sentence": "The Swans built on this, improving to finish third on the ladder with 16 wins and six losses",
                    "refs": []
                },
                {
                    "sentence": "They defeated reigning premiers Melbourne by 22 points in the second qualifying final, then in the preliminary final withstood a late rally by Collingwood to win by one point, returning to the grand final for the first time since 2016",
                    "refs": []
                },
                {
                    "sentence": "The two teams met only once in the home-and-away season, in Round 2 at the Sydney Cricket Ground",
                    "refs": []
                },
                {
                    "sentence": "In a game made memorable by Lance Franklin's 1000th career goal, Sydney won by 30 points",
                    "refs": []
                },
                {
                    "sentence": "This was the sixth time the teams had met in VFL/AFL finals, including twice before South Melbourne's relocation to Sydney, and the first time in a grand final",
                    "refs": []
                },
                {
                    "sentence": "Their most recent finals encounter was in a 2017 semi-final, which was Geelong's only previous finals victory against the Swans",
                    "refs": []
                },
                {
                    "sentence": "Prior to the formation of the VFL/AFL, the clubs had also previously met in the Victorian Football Association's \"Match of the Century\", which had decided the 1886 premiership in Geelong's favour.",
                    "refs": [
                        "https://web.archive.org/web/20170913000513/http://www.afl.com.au/match-centre/2017/25/geel-v-syd"
                    ]
                },
                {
                    "sentence": "Both clubs entered the game in strong form and on long winning streaks, Geelong having won its last fifteen games, and Sydney having won its last nine games",
                    "refs": []
                },
                {
                    "sentence": "Geelong was the strong favourite, the TAB offering odds of $1.47 for a Geelong victory against Sydney's $2.70 on game day",
                    "refs": [
                        "https://www.codesports.com.au/bet/afl/tips/geelongsydney-2022-afl-grand-final-what-the-firstlook-odds-tell-us/"
                    ]
                },
                {
                    "sentence": "The Melbourne Cricket Ground hosted the grand final for the first time since 2019, with the 2020 and 2021 editions previously being held at Brisbane's The Gabba and Perth's Optus Stadium respectively, due to the COVID-19 pandemic",
                    "refs": []
                },
                {
                    "sentence": "The capacity crowd of 100,024 was the largest at a VFL/AFL game since 1986.",
                    "refs": [
                        "https://www.theguardian.com/sport/2022/sep/24/geelong-eviscerate-sydney-swans-by-81-points-in-afl-grand-final-win-for-the-ages"
                    ]
                }
            ],
            "subsections": []
        },
        {
            "section_title": "Ceremonies and entertainment",
            "section_content": [],
            "subsections": [
                {
                    "section_title": "Parade",
                    "section_content": [
                        {
                            "sentence": "The annual Grand Final Parade returned to Melbourne for the first time since 2019 on the Friday before the Grand Final",
                            "refs": []
                        },
                        {
                            "sentence": "The novel parade included the players travelling along the Yarra River on barges before the players boarded utes and travelled through Yarra Park; their journey concluded north of the Melbourne Cricket Ground",
                            "refs": []
                        },
                        {
                            "sentence": "The Yarra River portion of the parade was criticised by many due to bad viewing angles from the banks of the river as well as the boats turning around before reaching Princes Bridge, where many fans had gathered in order to have a good view",
                            "refs": [
                                "https://www.abc.net.au/news/2022-09-23/grand-final-parade-route-afl-geelong-cats-sydney-swans/101456510"
                            ]
                        }
                    ],
                    "subsections": []
                },
                {
                    "section_title": "On-field events",
                    "section_content": [
                        {
                            "sentence": "All times are in Australian Eastern Standard Time (GMT +10)",
                            "refs": []
                        },
                        {
                            "sentence": "The Premiership Cup was brought onto the ground by Cameron Ling, a member of Geelong's last three premiership teams in 2007, 2009 and 2011, and former Sydney captain Paul Kelly",
                            "refs": []
                        },
                        {
                            "sentence": "Ling presented coach Chris Scott and captain Joel Selwood the cup after the match.",
                            "refs": []
                        },
                        {
                            "sentence": "Selwood carried Levi Ablett through Geelong's banner as the Cats took the field",
                            "refs": []
                        },
                        {
                            "sentence": "Levi is the son of Gary Ablett Jr., who attended the game having been a dual premiership player with Geelong in 2007 and 2009 when he also won the Brownlow Medal, and grandson of Cats and Victoria forward Gary Ablett Sr.; Levi had been diagnosed with a rare degenerative disease that left him mute",
                            "refs": []
                        }
                    ],
                    "subsections": []
                }
            ]
        },
        {
            "section_title": "Match summary",
            "section_content": [],
            "subsections": [
                {
                    "section_title": "First quarter",
                    "section_content": [
                        {
                            "sentence": "Geelong veteran Tom Hawkins kicked the first behind in the opening minute of the match",
                            "refs": []
                        },
                        {
                            "sentence": "Geelong dominated play and territory through the first ten minutes while Sydney managed to hold them out from scoring again, until the tenth minute, when Hawkins snatched the ball straight out of a boundary throw-in ruck contest and snapped the opening goal",
                            "refs": []
                        },
                        {
                            "sentence": "Five minutes later, Hawkins scored his second goal in exactly the same way, and Geelong led by 13 points",
                            "refs": []
                        },
                        {
                            "sentence": "Sydney won the ensuing centre clearance, ending with Will Hayward scored Sydney's first goal from a crumbing snap shot in the 17th minute",
                            "refs": [
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632e927f8f0822acf24f5b52&filterKeyEvents=false"
                            ]
                        },
                        {
                            "sentence": "Thereafter, Geelong dominated the quarter",
                            "refs": [
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632e927f8f0822acf24f5b52&filterKeyEvents=false"
                            ]
                        },
                        {
                            "sentence": "The Cats kicked three goals in the next five minutes: Mark Blicavs (19th minute) with a set shot from 25m, and Isaac Smith (20th minute and 22nd minute) with two crumbed goals on the run",
                            "refs": []
                        },
                        {
                            "sentence": "Geelong generated six more shots in the quarter, for 1.4 (10), only Brad Close managing to kick a goal from a 40m set shot in the 27th minute",
                            "refs": []
                        },
                        {
                            "sentence": "When quarter time sounded, Geelong led 6.5 (41) to Sydney's 1.0 (6)",
                            "refs": []
                        },
                        {
                            "sentence": "Geelong's score and leading margin at that point were both the largest any team had achieved in a grand final since the 1989",
                            "refs": []
                        },
                        {
                            "sentence": "All statistics skewed heavily in Geelong's favour, with a 48\u201329 advantage in contested possessions, an inside-50 tally of 20\u20138, and triple Sydney's marks",
                            "refs": []
                        },
                        {
                            "sentence": "Joel Selwood was Geelong's leading disposal winner, with 12;  while the tactical decision to play Sam De Koning as a loose defender drifting onto the wing had been particularly influential in Geelong's effectively preventing Sydney from advancing the ball",
                            "refs": [
                                "https://www.heraldsun.com.au/sport/afl/mick-malthouse-how-geelong-broke-down-sydney-in-grand-final/news-story/d288301167d6d1f5eb03b1b3f4c54211",
                                "https://www.theguardian.com/sport/2022/sep/24/geelong-eviscerate-sydney-swans-by-81-points-in-afl-grand-final-win-for-the-ages"
                            ]
                        }
                    ],
                    "subsections": []
                },
                {
                    "section_title": "Second quarter",
                    "section_content": [
                        {
                            "sentence": "Sydney won the opening centre clearance, ending with a behind to Lance Franklin",
                            "refs": []
                        },
                        {
                            "sentence": "But, Geelong rebounded and again dominated the territory, at one stage enjoying a run of recording 18 out of 19 consecutive inside-50s",
                            "refs": []
                        },
                        {
                            "sentence": "Over the next eight minutes, Geelong managed 1.2 (8), Tyson Stengle kicking the goal from a 50m set shot in the 5th minute",
                            "refs": []
                        },
                        {
                            "sentence": "Thereafter, Sydney's competitiveness improved, and the balance of the quarter was much more even, with many repeat stoppages and ruck contests, and Sydney's run-and-carry game getting started",
                            "refs": []
                        },
                        {
                            "sentence": "After winning the ball from a free kick, Sydney kicked deep to Hayden McLean who marked one-handed in the goal square and kicked Sydney's second goal in the 9th minute",
                            "refs": [
                                "https://www.afl.com.au/afl/matches/4773"
                            ]
                        },
                        {
                            "sentence": "In the 17th minute, Hawkins won a free kick in a one-on-one marking contest and kicked his third goal",
                            "refs": []
                        },
                        {
                            "sentence": "In the 20th minute, Callum Mills kicked a long goal for Sydney from 50m",
                            "refs": []
                        },
                        {
                            "sentence": "The ensuing centre clearance was won by Geelong and ended with a mark to Stengle, who kicked his second goal from a 40m set shot",
                            "refs": []
                        },
                        {
                            "sentence": "After turning over a kick-in, Isaac Heeney kicked a goal from a tight angle for Sydney in the 27th minute",
                            "refs": []
                        },
                        {
                            "sentence": "It was the last score of the quarter, and at half time Geelong had extended its quarter time lead by one point, Geelong 9.8 (62) to Sydney 4.2 (26)",
                            "refs": []
                        },
                        {
                            "sentence": "Patrick Dangerfield was a dominant force in the centre for Geelong through the first half, winning several centre clearances which set up Geelong scoring opportunities",
                            "refs": []
                        }
                    ],
                    "subsections": []
                },
                {
                    "section_title": "Third quarter",
                    "section_content": [
                        {
                            "sentence": "Geelong opened the third quarter with three goals in six minutes: the first went to Mitch Duncan from a 15m set shot in the 2nd minute, after winning a holding the ball free kick against Tom McCartin; the second went to Close in the 4th minute from a 15m set shot, after he intercepted a poor short pass across the face of goal by Tom McCartin; and the third was by Smith from a 50m shot",
                            "refs": []
                        },
                        {
                            "sentence": "This opening flurry of goals extended Geelong's advantage to 54 points, all but securing the premiership",
                            "refs": []
                        },
                        {
                            "sentence": "During this period, Sydney's Sam Reid was substituted out of the game and Braeden Campbell came on",
                            "refs": [
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632ea2f58f0822acf24f5b8d&filterKeyEvents=false"
                            ]
                        },
                        {
                            "sentence": "Sydney's only shot on goal for the quarter came in the tenth minute, and was rushed through for a behind",
                            "refs": [
                                "https://www.afl.com.au/news/850185/swans-decision-to-play-key-forward-costly-as-reid-subbed-out"
                            ]
                        },
                        {
                            "sentence": "Geelong continued to dominate, adding a further 3.3 (21), with goals to Cam Guthrie from a holding the ball free kick in the 12th minute, and to Stengle in the 18th and 21st minutes.",
                            "refs": []
                        },
                        {
                            "sentence": "At three quarter time, Geelong's lead was an insurmountable 74 points: Geelong 15.11 (101) vs Sydney 4.3 (27)",
                            "refs": []
                        },
                        {
                            "sentence": "Geelong once again completely dominated territory (71% time in forward half)  and contested ball for the quarter",
                            "refs": [
                                "https://www.afl.com.au/afl/matches/4773"
                            ]
                        },
                        {
                            "sentence": "Dangerfield contributed significantly to Geelong's drive, with ten disposals and three clearances",
                            "refs": [
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632ea2f58f0822acf24f5b8d&filterKeyEvents=false",
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?filterKeyEvents=false",
                                "https://afltables.com/afl/stats/games/2022/091620220924.html"
                            ]
                        }
                    ],
                    "subsections": []
                },
                {
                    "section_title": "Final quarter",
                    "section_content": [
                        {
                            "sentence": "With the result beyond doubt, the final quarter was played with lesser intensity but still at a lively pace, Geelong kicking five goals to Sydney's four",
                            "refs": []
                        },
                        {
                            "sentence": "Goals were scored by Sydney's Chad Warner (6th minute); Geelong's Jeremy Cameron (7th minute); Geelong's Brandan Parfitt, who was substituted into the game for Cam Guthrie in the 15th minute, then crumbed a goal in the 16th minute; Sydney's Paddy McCartin (17th minute); De Koning (20th minute), who marked a Dangerfield snap shot on the goal line and kicked the first goal of his 24-game career; Warner (22nd minute) with a 70m long bomb; Geelong captain Joel Selwood (25th minute); Cameron (27th minute); and Sydney's Tom Papley (30th minute)",
                            "refs": []
                        },
                        {
                            "sentence": "The final score was Geelong 20.13 (133), Sydney 8.4 (52)",
                            "refs": [
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?filterKeyEvents=false"
                            ]
                        }
                    ],
                    "subsections": []
                },
                {
                    "section_title": "Overall",
                    "section_content": [
                        {
                            "sentence": "Geelong had a statistical dominance throughout the game, particularly in the first and third quarters",
                            "refs": []
                        },
                        {
                            "sentence": "Most starkly demonstrating Geelong's advantage it won: the inside-50s count 63\u201332; contested possessions 151\u2013110; total disposals 395\u2013304;  time in forward half 65%\u201335%",
                            "refs": [
                                "https://afltables.com/afl/stats/games/2022/091620220924.html"
                            ]
                        },
                        {
                            "sentence": "Sydney won slightly more clearances (37\u201333),  but Geelong's clearances were won more cleanly,  such that Geelong lost possession after only one of its clearances compared with Sydney's 17, and led scores from stoppages 65\u201315",
                            "refs": [
                                "https://www.afl.com.au/afl/matches/4773",
                                "https://afltables.com/afl/stats/games/2022/091620220924.html",
                                "https://www.heraldsun.com.au/sport/afl/mick-malthouse-how-geelong-broke-down-sydney-in-grand-final/news-story/d288301167d6d1f5eb03b1b3f4c54211"
                            ]
                        },
                        {
                            "sentence": "Sydney's rebound game was completely nullified by Geelong's strong marking and ground ball contests through its half-forward and centre-line, allowing Geelong repeat forward 50 entries throughout the game",
                            "refs": []
                        }
                    ],
                    "subsections": []
                },
                {
                    "section_title": "Norm Smith Medal",
                    "section_content": [
                        {
                            "sentence": "Geelong winger Isaac Smith, who had a game-high 32 disposals, a game-high 772 metres gained on the ground, three goals and 14 total score involvements to be responsible for much of Geelong's attacking play, received the Norm Smith Medal, polling 14 out of a possible 15 votes",
                            "refs": []
                        },
                        {
                            "sentence": "At age 33, he was the oldest recipient of the accolade in its history",
                            "refs": []
                        },
                        {
                            "sentence": "2002 Norm Smith Medallist Nathan Buckley presented the medal.",
                            "refs": [
                                "https://www.abc.net.au/news/2022-09-24/isaac-smith-norm-smith-medal-patrick-dangerfield-geelong-afl/101471748"
                            ]
                        },
                        {
                            "sentence": "Midfielder Patrick Dangerfield was second with 10 votes; his nine clearances \u2013 many won cleanly and at speed from centre bounces \u2013 and 19 contested possessions launched many Geelong scores from stoppages",
                            "refs": []
                        },
                        {
                            "sentence": "Small forward Tyson Stengle, who was the leading goalkicker with 4.1, polled four votes",
                            "refs": [
                                "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?filterKeyEvents=false"
                            ]
                        },
                        {
                            "sentence": "Tall defender Sam De Koning, whose work intercepting Sydney rebounds in the centre of the ground contributed significantly to Geelong's territorial advantage; and ruck/midfielder Mark Blicavs, with 14 contested possessions, 15 hit-outs and a team-high eight tackles \u2013 each polled one vote",
                            "refs": []
                        },
                        {
                            "sentence": "Among the other top performing Geelong players who did not poll votes were Tom Hawkins (scored 3.4, including the opening two goals), Joel Selwood (26 disposals), Rhys Stanley (27 hitouts), Jake Kolodjashnij (nullified Sydney's Isaac Heeney), Mitch Duncan (27 disposals), Brad Close (18 disposals, two goals) and Cameron Guthrie (16 disposals)",
                            "refs": []
                        },
                        {
                            "sentence": "Very few Sydney players were considered to have had good games: midfielders Chad Warner (29 disposals, 10 clearances) and James Rowbottom (16 disposals, 8 clearances) and defender Robbie Fox (nullified Jeremy Cameron until the final quarter) considered the Swans' best",
                            "refs": []
                        }
                    ],
                    "subsections": []
                }
            ]
        },
        {
            "section_title": "Teams",
            "section_content": [
                {
                    "sentence": "Geelong's starting 22 made one change from its preliminary final team",
                    "refs": []
                },
                {
                    "sentence": "Wingman Max Holmes, who was substituted out of the preliminary final with a hamstring injury, was included in Thursday's selected team, but was omitted on the morning of the game; he was replaced in the 22 by preliminary final medical substitute Mark O\u2019Connor, and Brandan Parfitt came into the 23 as substitute",
                    "refs": []
                },
                {
                    "sentence": "Playing his 40th final, Geelong captain Joel Selwood surpassed Michael Tuck's record for most career VFL/AFL finals,  and he became Geelong's only four-time VFL/AFL premiership player",
                    "refs": [
                        "https://www.aap.com.au/news/oconnor-called-into-cats-afl-gf-team/",
                        "https://afltables.com/afl/stats/alltime/misc_players.html#09"
                    ]
                },
                {
                    "sentence": "The grand final was ultimately Selwood's last game, as he retired less than a week later",
                    "refs": []
                },
                {
                    "sentence": "With an average age of 28 years and 206 days, Geelong's grand final 23 set the record for the oldest selected team in any VFL/AFL game, grand final or otherwise, 67 days older than any previous team",
                    "refs": []
                },
                {
                    "sentence": "Zach Tuohy and Mark O'Connor became only the second and third Irish players to win an AFL title, after Tadhg Kennelly in 2005",
                    "refs": [
                        "https://afltables.com/afl/stats/alltime/misc.html#age"
                    ]
                },
                {
                    "sentence": "Sydney made one change to its preliminary final 22, with Logan McDonald dropped after playing 17 games in 2022 for Hayden McLean, who played his first game at the AFL level since Round 8",
                    "refs": []
                },
                {
                    "sentence": "Preliminary final medical substitute Braeden Campbell held his place in the 23",
                    "refs": []
                },
                {
                    "sentence": "Two Swans under injury clouds \u2014 Sam Reid, who had been substituted out of the preliminary final with an adductor strain, and Justin McInerney, who had an ankle complaint \u2014 were both selected",
                    "refs": []
                },
                {
                    "sentence": "Reid played only half of the grand final, with limited impact, and was substituted out for Campbell shortly after half time with a recurrence of the same injury \u2013 and Sydney's decision to select him was widely criticised by commentators",
                    "refs": [
                        "https://www.aap.com.au/news/oconnor-called-into-cats-afl-gf-team/"
                    ]
                }
            ],
            "subsections": [
                {
                    "section_title": "Umpires",
                    "section_content": [
                        {
                            "sentence": "The umpiring panel, comprising three field umpires, four boundary umpires, two goal umpires and an emergency in each position, is given below",
                            "refs": []
                        },
                        {
                            "sentence": "1977 VFL Grand Final umpire John Sutcliffe presented the umpires' medals in the post-match ceremony",
                            "refs": []
                        },
                        {
                            "sentence": "Numbers in brackets represent the number of grand finals umpired, including 2022.",
                            "refs": []
                        }
                    ],
                    "subsections": []
                }
            ]
        },
        {
            "section_title": "Scoreboard",
            "section_content": [],
            "subsections": []
        },
        {
            "section_title": "Media coverage",
            "section_content": [],
            "subsections": [
                {
                    "section_title": "Television",
                    "section_content": [
                        {
                            "sentence": "Per the AFL TV rights, and for the eleventh consecutive year, the Seven Network had exclusive broadcast rights within Australia, with Fox Footy showing replays after the game's conclusion",
                            "refs": []
                        },
                        {
                            "sentence": "The Seven coverage was led by James Brayshaw and Brian Taylor, with Luke Hodge and Daisy Pearce providing special comments",
                            "refs": []
                        },
                        {
                            "sentence": "Abbey Holmes and Matthew Richardson provided interviews at the breaks, and boundary updates throughout the match",
                            "refs": []
                        },
                        {
                            "sentence": "Hamish McLachlan served as Master of Ceremonies for the match as well as hosting the Seven coverage alongside Brayshaw",
                            "refs": []
                        },
                        {
                            "sentence": "The Seven coverage was simulcast on 7plus on select devices for the first time",
                            "refs": []
                        },
                        {
                            "sentence": "Fox showed their own coverage with their own team, which started at 9\u00a0am AEST with events such as the Longest Kick across the Yarra River and pre-match, half-time and post-match analysis",
                            "refs": []
                        },
                        {
                            "sentence": "The Fox Footy coverage was simulcast on Kayo Sports",
                            "refs": []
                        },
                        {
                            "sentence": "Sarah Jones, Garry Lyon and Kath Loughnan led the Fox broadcast with special inputs from Brad Johnson, Cameron Mooney, David King, Jonathan Brown, Jordan Lewis, Leigh Montagna, Jason Dunstall, Ben Dixon, Nathan Buckley and Nick Riewoldt throughout the coverage",
                            "refs": []
                        },
                        {
                            "sentence": "TV ratings for the game were comparatively low for a Grand Final",
                            "refs": []
                        },
                        {
                            "sentence": "An average of 2.18\u00a0million viewers in capital cities (plus 95,000 viewers on streaming platforms) were recorded making it the lowest-rating broadcast of modern history,  behind only the 1993 AFL Grand Final between Essendon and Carlton, although viewership peaked at 5.76\u00a0million",
                            "refs": [
                                "https://mumbrella.com.au/afl-grand-final-lowest-metro-audience-since-2001-nine-steals-the-week-757737/amp"
                            ]
                        },
                        {
                            "sentence": "Seven's broadcast of the game via its streaming services, where its rights were non-exclusive, also experienced some technical problems",
                            "refs": [
                                "https://www.theage.com.au/sport/afl/seven-renews-push-for-night-grand-final-after-ratings-slump-for-one-sided-decider-20220925-p5bkts.html"
                            ]
                        },
                        {
                            "sentence": "Despite the lowest TV ratings in 29 years, it still received more than a million more viewers than any other non-AFL TV show of the day, and it still retained a 65.8% network share as well as the distinction of being most-watched TV show of the year",
                            "refs": [
                                "https://tvtonight.com.au/2022/09/afl-fants-vent-as-7plus-apologises-for-technical-difficulties.html",
                                "https://www.afr.com/companies/sport/afl-grand-final-a-ratings-bomb-for-channel-7-20220925-p5bkt9"
                            ]
                        },
                        {
                            "sentence": "Prior to the game, AFL CEO Gillon McLachlan predicted the game would get 4.4\u00a0million viewers, more than 1.3\u00a0million more than the actual figure",
                            "refs": [
                                "https://tvtonight.com.au/tag/afl-grand-final"
                            ]
                        },
                        {
                            "sentence": "Channel Seven boss James Warburton called for the match to be brought back to a twilight or night time slot in order to increase ratings",
                            "refs": [
                                "https://www.theage.com.au/sport/afl/grand-final-ratings-record-on-cards-after-buddy-signing-boosts-swans-20220920-p5bjk7.html"
                            ]
                        }
                    ],
                    "subsections": [
                        {
                            "section_title": "International",
                            "section_content": [],
                            "subsections": []
                        }
                    ]
                },
                {
                    "section_title": "Radio",
                    "section_content": [],
                    "subsections": []
                }
            ]
        },
        {
            "section_title": "References",
            "section_content": [],
            "subsections": []
        },
        {
            "section_title": "External links",
            "section_content": [],
            "subsections": []
        }
    ],
    "references": {
        "1": "https://www.theguardian.com/sport/2022/sep/24/geelong-eviscerate-sydney-swans-by-81-points-in-afl-grand-final-win-for-the-ages",
        "2": "http://www.afl.com.au/tv-radio/broadcastguide",
        "3": "https://www.afl.com.au/afl/matches/4741#match-report",
        "4": "https://www.afl.com.au/afl/matches/4767#match-report",
        "5": "https://www.afl.com.au/news/833821/sublime-swans-stun-dees-to-earn-home-prelim-final",
        "6": "https://www.afl.com.au/afl/matches/4768#match-report",
        "7": "https://www.afl.com.au/news/847462/six-of-the-best-swamps-grand-final-stats-spectacular",
        "8": "https://www.afl.com.au/ladder?Competition=1&CompSeason=43&GameWeeks=610&ShowBettingOdds=1&Live=0",
        "9": "https://web.archive.org/web/20170913000513/http://www.afl.com.au/match-centre/2017/25/geel-v-syd",
        "10": "https://www.codesports.com.au/bet/afl/tips/geelongsydney-2022-afl-grand-final-what-the-firstlook-odds-tell-us/",
        "11": "https://www.heraldsun.com.au/sport/afl/afl-2022-champion-data-takes-a-deep-dive-into-the-grand-final-between-geelong-and-sydney/news-story/c1570de8475da0a31eef93a08c3f5948",
        "12": "https://www.abc.net.au/news/2022-09-23/grand-final-parade-route-afl-geelong-cats-sydney-swans/101456510",
        "13": "https://www.9news.com.au/national/afl-grand-final-fans-disappointed-by-the-return-of-melbournes-afl-grand-final-parade/90def24a-330f-47fd-be9e-5695380caa30",
        "14": "https://www.heraldsun.com.au/sport/afl/angry-afl-fans-unleash-on-grand-final-parade-failure/news-story/53c674476a14cfea5a24c6fe0bd8a32f",
        "15": "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632e74f68f0822acf24f5aae#block-632e74f68f0822acf24f5aae",
        "16": "https://www.afl.com.au/news/848092/delta-goodrem-joins-the-2022-telstra-pre-match-entertainment-as-special-guest",
        "17": "https://7news.com.au/sport/afl/beautiful-scenes-as-levi-ablett-runs-through-afl-grand-final-banner-with-geelong-c-8341686",
        "18": "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632e927f8f0822acf24f5b52&filterKeyEvents=false",
        "19": "https://afltables.com/afl/stats/games/2022/091620220924.html",
        "20": "https://www.heraldsun.com.au/sport/afl/mick-malthouse-how-geelong-broke-down-sydney-in-grand-final/news-story/d288301167d6d1f5eb03b1b3f4c54211",
        "21": "https://www.afl.com.au/afl/matches/4773",
        "22": "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632ea2f58f0822acf24f5b8d&filterKeyEvents=false",
        "23": "https://www.afl.com.au/news/850185/swans-decision-to-play-key-forward-costly-as-reid-subbed-out",
        "24": "https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?filterKeyEvents=false",
        "25": "https://www.abc.net.au/news/2022-09-24/isaac-smith-norm-smith-medal-patrick-dangerfield-geelong-afl/101471748",
        "26": "https://web.archive.org/web/20220924215331/https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632eb3638f0894f47d8a4955%23block-632eb3638f0894f47d8a4955",
        "27": "https://www.heraldsun.com.au/sport/afl/afl-grand-final-2022-geelong-defeats-sydney-by-81-points-player-ratings/news-story/2f17962e795bda8fdf02fb2624c25639",
        "28": "https://www.aap.com.au/news/oconnor-called-into-cats-afl-gf-team/",
        "29": "https://afltables.com/afl/stats/alltime/misc_players.html#09",
        "30": "https://afltables.com/afl/stats/alltime/misc.html#age",
        "31": "https://www.rte.ie/sport/other-sport/2022/0924/1325214-grand-final-joy-for-irish-duo-as-geelong-destroy-sydney/",
        "32": "https://www.afl.com.au/news/850185/swans-decision-to-play-key-forward-costly-as-reid-subbed-out",
        "33": "https://www.geelongcats.com.au/news/1227026/grand-final-preview-everything-you-need-to-know-before-the-big-one",
        "34": "https://web.archive.org/web/20220924210014/https://www.theguardian.com/sport/live/2022/sep/24/afl-grand-final-2022-geelong-cats-vs-sydney-swans-live-scores-winners-win-team-news-time-start-entertainment-updates-teams-mcg?page=with:block-632e82008f0894f47d8a486b%23block-632e82008f0894f47d8a486b",
        "35": "https://www.theroar.com.au/2022/09/24/how-do-you-watch-the-2022-grand-final-afl-broadcast-and-live-streaming-guide/",
        "36": "https://mumbrella.com.au/afl-grand-final-lowest-metro-audience-since-2001-nine-steals-the-week-757737/amp",
        "37": "https://www.theage.com.au/sport/afl/seven-renews-push-for-night-grand-final-after-ratings-slump-for-one-sided-decider-20220925-p5bkts.html",
        "38": "https://tvtonight.com.au/2022/09/afl-fants-vent-as-7plus-apologises-for-technical-difficulties.html",
        "39": "https://www.afr.com/companies/sport/afl-grand-final-a-ratings-bomb-for-channel-7-20220925-p5bkt9",
        "40": "https://tvtonight.com.au/tag/afl-grand-final",
        "41": "https://www.theage.com.au/sport/afl/grand-final-ratings-record-on-cards-after-buddy-signing-boosts-swans-20220920-p5bjk7.html",
        "42": "https://www.theage.com.au/sport/afl/seven-renews-push-for-night-grand-final-after-ratings-slump-for-one-sided-decider-20220925-p5bkts.html",
        "43": "https://www.theroar.com.au/2022/09/23/how-to-watch-the-2022-afl-grand-final-overseas-geelong-cats-vs-sydney-swans-international-live-stream-info/",
        "44": "https://www.afl.com.au/matches/broadcast-guide/international-broadcast-partners"
    },
    "spicy_entities": {
        "2022": "DATE",
        "australian": "NORP",
        "geelong": "ORG",
        "the sydney swans": "ORG",
        "the melbourne cricket ground": "ORG",
        "24 september 2022": "DATE",
        "127th": "DATE",
        "the australian football league": "ORG",
        "100,024": "CARDINAL",
        "81": "CARDINAL",
        "tenth": "ORDINAL",
        "vfl": "ORG",
        "isaac smith": "PERSON",
        "the norm smith medal": "ORG",
        "2021": "DATE",
        "melbourne": "GPE",
        "early in the season": "DATE",
        "five": "CARDINAL",
        "four": "CARDINAL",
        "second": "ORDINAL",
        "four seasons": "DATE",
        "18": "CARDINAL",
        "collingwood": "PERSON",
        "six": "CARDINAL",
        "the brisbane lions": "ORG",
        "71": "CARDINAL",
        "first": "ORDINAL",
        "three years": "DATE",
        "sydney": "GPE",
        "one": "CARDINAL",
        "greater western": "LOC",
        "swans": "NORP",
        "third": "ORDINAL",
        "16": "CARDINAL",
        "22": "CARDINAL",
        "2016": "DATE",
        "two": "CARDINAL",
        "round 2": "LAW",
        "the sydney cricket ground": "ORG",
        "lance franklin's": "ORG",
        "1000th": "ORDINAL",
        "30": "CARDINAL",
        "sixth": "ORDINAL",
        "2017": "DATE",
        "the victorian football association's": "ORG",
        "match of the century": "WORK_OF_ART",
        "1886": "DATE",
        "fifteen": "CARDINAL",
        "nine": "CARDINAL",
        "tab": "ORG",
        "1.47": "MONEY",
        "2.70": "MONEY",
        "2019": "DATE",
        "2020": "DATE",
        "brisbane's the gabba": "ORG",
        "perth": "GPE",
        "optus stadium": "FAC",
        "covid-19": "ORG",
        "1986": "DATE",
        "annual": "DATE",
        "grand final parade": "EVENT",
        "friday": "DATE",
        "the grand final": "ORG",
        "the yarra river": "LOC",
        "yarra park": "GPE",
        "yarra river": "LOC",
        "princes bridge": "FAC",
        "gmt": "ORG",
        "the premiership cup": "EVENT",
        "cameron ling": "PERSON",
        "three": "CARDINAL",
        "2007": "DATE",
        "2009": "DATE",
        "2011": "DATE",
        "paul kelly": "PERSON",
        "chris scott": "PERSON",
        "joel selwood": "PERSON",
        "levi ablett": "PERSON",
        "cats": "ORG",
        "levi": "PERSON",
        "gary ablett jr.": "PERSON",
        "the brownlow medal": "WORK_OF_ART",
        "victoria": "GPE",
        "gary ablett sr.": "PERSON",
        "first quarter": "DATE",
        "tom hawkins": "PERSON",
        "the opening minute": "TIME",
        "the first ten minutes": "TIME",
        "the tenth minute": "TIME",
        "hawkins": "PERSON",
        "five minutes later": "TIME",
        "13": "CARDINAL",
        "will hayward": "PERSON",
        "the quarter": "DATE",
        "the next five minutes": "TIME",
        "mark blicavs": "PERSON",
        "19th minute": "TIME",
        "25": "CARDINAL",
        "20th minute": "TIME",
        "22nd minute": "TIME",
        "1.4": "CARDINAL",
        "10": "CARDINAL",
        "brad close": "PERSON",
        "40": "CARDINAL",
        "27th minute": "TIME",
        "quarter": "DATE",
        "6.5": "CARDINAL",
        "41": "CARDINAL",
        "1.0": "CARDINAL",
        "6": "CARDINAL",
        "1989": "DATE",
        "inside-50": "DATE",
        "20\u20138": "GPE",
        "12": "CARDINAL",
        "sam de koning": "PERSON",
        "second quarter": "DATE",
        "lance franklin": "ORG",
        "19 consecutive": "DATE",
        "the next eight minutes": "TIME",
        "1.2": "CARDINAL",
        "8)": "DATE",
        "tyson stengle": "PERSON",
        "50": "CARDINAL",
        "5th": "ORDINAL",
        "hayden mclean": "PERSON",
        "9th": "ORDINAL",
        "20th": "ORDINAL",
        "callum mills": "PERSON",
        "stengle": "PERSON",
        "isaac heeney": "PERSON",
        "half": "CARDINAL",
        "geelong 9.8": "ORG",
        "62": "CARDINAL",
        "26": "CARDINAL",
        "patrick dangerfield": "PERSON",
        "the first half": "DATE",
        "third quarter": "DATE",
        "the third quarter": "DATE",
        "six minutes": "TIME",
        "mitch duncan": "PERSON",
        "15": "CARDINAL",
        "tom mccartin": "PERSON",
        "4th": "ORDINAL",
        "smith": "GPE",
        "54": "CARDINAL",
        "sam reid": "PERSON",
        "braeden campbell": "PERSON",
        "3.3": "CARDINAL",
        "21": "CARDINAL",
        "cam guthrie": "PERSON",
        "the 18th and 21st": "DATE",
        "three quarter": "DATE",
        "74": "CARDINAL",
        "15.11": "CARDINAL",
        "101": "CARDINAL",
        "sydney 4.3": "PERSON",
        "27": "CARDINAL",
        "71%": "PERCENT",
        "ten": "CARDINAL",
        "the final quarter": "DATE",
        "chad warner": "PERSON",
        "6th minute": "TIME",
        "jeremy cameron": "PERSON",
        "7th minute": "TIME",
        "brandan parfitt": "PERSON",
        "15th": "ORDINAL",
        "16th": "ORDINAL",
        "paddy mccartin": "PERSON",
        "17th minute": "TIME",
        "de koning": "ORG",
        "dangerfield": "ORG",
        "24": "CARDINAL",
        "warner": "ORG",
        "70": "CARDINAL",
        "25th minute": "TIME",
        "cameron": "PERSON",
        "tom papley": "PERSON",
        "30th minute": "TIME",
        "20.13": "CARDINAL",
        "133": "CARDINAL",
        "8.4": "CARDINAL",
        "52": "CARDINAL",
        "the first and third quarters": "DATE",
        "inside-50s": "LOC",
        "63\u201332": "CARDINAL",
        "151\u2013110": "CARDINAL",
        "395\u2013304": "CARDINAL",
        "65%\u201335%": "PERCENT",
        "only one": "CARDINAL",
        "17": "CARDINAL",
        "65\u201315": "CARDINAL",
        "norm smith medal": "PERSON",
        "32": "CARDINAL",
        "772 metres": "QUANTITY",
        "14": "CARDINAL",
        "age 33": "DATE",
        "2002": "DATE",
        "norm smith": "PERSON",
        "nathan buckley": "PERSON",
        "19": "CARDINAL",
        "4.1": "CARDINAL",
        "eight": "CARDINAL",
        "3.4": "CARDINAL",
        "rhys stanley": "ORG",
        "jake kolodjashnij": "PERSON",
        "cameron guthrie": "PERSON",
        "29": "CARDINAL",
        "james rowbottom": "PERSON",
        "8": "CARDINAL",
        "robbie fox": "PERSON",
        "wingman max holmes": "PERSON",
        "thursday": "DATE",
        "the morning": "TIME",
        "mark o\u2019connor": "PERSON",
        "23": "CARDINAL",
        "40th": "ORDINAL",
        "michael tuck's": "PERSON",
        "only four": "CARDINAL",
        "selwood": "GPE",
        "less than a week later": "DATE",
        "206 days": "DATE",
        "67 days": "DATE",
        "zach tuohy": "PERSON",
        "mark o'connor": "PERSON",
        "irish": "NORP",
        "afl": "ORG",
        "tadhg kennelly": "PERSON",
        "2005": "DATE",
        "logan mcdonald": "PERSON",
        "round 8": "LAW",
        "justin mcinerney": "PERSON",
        "reid": "PERSON",
        "only half": "CARDINAL",
        "campbell": "ORG",
        "umpires": "PERSON",
        "1977": "DATE",
        "vfl grand final": "LAW",
        "john sutcliffe": "PERSON",
        "afl tv": "ORG",
        "the eleventh consecutive year": "DATE",
        "the seven network": "ORG",
        "australia": "GPE",
        "fox footy": "ORG",
        "seven": "CARDINAL",
        "james brayshaw": "PERSON",
        "brian taylor": "PERSON",
        "luke hodge": "PERSON",
        "daisy pearce": "PERSON",
        "abbey holmes": "PERSON",
        "matthew richardson": "PERSON",
        "hamish mclachlan": "PERSON",
        "master of ceremonies": "WORK_OF_ART",
        "brayshaw": "ORG",
        "7plus": "CARDINAL",
        "fox": "ORG",
        "9": "CARDINAL",
        "aest": "ORG",
        "kayo sports": "ORG",
        "sarah jones": "PERSON",
        "garry lyon": "PERSON",
        "kath loughnan": "PERSON",
        "brad johnson": "PERSON",
        "cameron mooney": "PERSON",
        "david king": "PERSON",
        "jonathan brown": "PERSON",
        "jordan lewis": "PERSON",
        "leigh montagna": "PERSON",
        "jason dunstall": "PERSON",
        "ben dixon": "PERSON",
        "nick riewoldt": "PERSON",
        "2.18\u00a0million": "CARDINAL",
        "95,000": "CARDINAL",
        "1993": "DATE",
        "essendon": "ORG",
        "carlton": "GPE",
        "5.76\u00a0million": "MONEY",
        "29 years": "DATE",
        "more than a million": "CARDINAL",
        "non-afl tv": "ORG",
        "the day": "DATE",
        "65.8%": "PERCENT",
        "gillon mclachlan": "PERSON",
        "4.4\u00a0million": "CARDINAL",
        "more than 1.3\u00a0million": "CARDINAL",
        "james warburton": "PERSON",
        "twilight or night": "TIME"
    },
    "flair_entities": [
        "brad close",
        "abbey holmes",
        "rhys stanley",
        "afl grand final",
        "grand final",
        "chris scott",
        "collingwood",
        "sydney",
        "levi ablett",
        "john sutcliffe",
        "patrick dangerfield",
        "brownlow medal",
        "sarah jones",
        "daisy pearce",
        "james rowbottom",
        "hamish mclachlan",
        "michael tuck",
        "yarra park",
        "norm smith medal  geelong",
        "brisbane lions",
        "essendon",
        "robbie fox",
        "isaac heeney",
        "fox footy",
        "braeden campbell",
        "australian football league",
        "chad warner",
        "smith",
        "optus stadium",
        "cameron mooney",
        "greater western sydney",
        "james warburton",
        "garry lyon",
        "kayo sports",
        "sam reid",
        "tyson stengle",
        "tom mccartin",
        "sam de koning",
        "ling",
        "kath loughnan",
        "premiership cup",
        "afl",
        "max holmes",
        "selwood",
        "paddy mccartin",
        "mark blicavs",
        "south melbourne",
        "joel selwood",
        "de koning",
        "geelong",
        "sydney swans",
        "princes bridge",
        "swans'",
        "warner",
        "brian taylor",
        "brad johnson",
        "jordan lewis",
        "dangerfield",
        "afl tv",
        "seven network",
        "lance franklin",
        "leigh montagna",
        "cameron guthrie",
        "david king",
        "luke hodge",
        "logan mcdonald",
        "the gabba",
        "tom papley",
        "jake kolodjashnij",
        "will hayward",
        "zach tuohy",
        "victorian football association",
        "australian eastern standard time",
        "carlton",
        "vfl",
        "round 2",
        "tom hawkins",
        "jason dunstall",
        "hayden mclean",
        "callum mills",
        "stengle",
        "matthew richardson",
        "brayshaw",
        "james brayshaw",
        "melbourne",
        "reid",
        "isaac smith",
        "gary ablett sr.",
        "norm smith",
        "longest kick",
        "hawkins",
        "ben dixon",
        "sydney cricket ground",
        "brisbane",
        "justin mcinerney",
        "australia",
        "7plus",
        "victoria",
        "jeremy cameron",
        "jonathan brown",
        "mitch duncan",
        "grand final parade",
        "swans",
        "paul kelly",
        "cam guthrie",
        "gary ablett jr.",
        "perth",
        "nick riewoldt",
        "levi",
        "gillon mclachlan",
        "yarra river",
        "fox",
        "norm smith medal",
        "australian",
        "gmt",
        "melbourne cricket ground",
        "close",
        "tab",
        "covid-19",
        "nathan buckley",
        "channel seven",
        "irish",
        "vfl grand final",
        "cameron ling",
        "mark o\u2019connor",
        "brandan parfitt",
        "campbell",
        "cats",
        "mark o'connor",
        "cameron",
        "tadhg kennelly"
    ]
}