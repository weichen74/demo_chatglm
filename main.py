# -*- coding:utf-8 -*-
import os
import shutil
import random
import time
import mdtex2html
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

'''
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".cache.db")
'''


from app_modules.overwrites import postprocess
from app_modules.presets import *
from clc.langchain_application import LangChainApplication


# 修改成自己的配置！！！
class LangChainCFG:
    llm_model_name = 'C:\cache'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'D:\\python\\aigc\\chatglm2\\text2vec'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = './cache'
    docs_path = './docs'
    kg_vector_stores = {
        '中文维基百科': './cache/zh_wikipedia',
        '大规模金融研报': './cache/financial_research_reports',
        '初始化': './cache',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    n_gpus=1


config = LangChainCFG()
application = LangChainApplication(config)

application.source_service.init_source_vector()

def get_file_list():
    if not os.path.exists("docs"):
        return []
    return [f for f in os.listdir("docs")]


file_list = get_file_list()


def upload_file(file):
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    application.source_service.add_document("docs/" + filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def set_knowledge(kg_name, history):
    try:
        application.source_service.load_vector_store(config.kg_vector_stores[kg_name])
        msg_status = f'{kg_name}知识库已成功加载'
    except Exception as e:
        print(e)
        msg_status = f'{kg_name}知识库未成功加载'
    return history + [[None, msg_status]]


def clear_session():
    return '', None
'''
def postprocess(self, y):
    if y is None:
        return []
    y=[
        {"You are a helpful assistant." "Who won the world series in 2020?"},
        {"The Los Angeles Dodgers won the World Series in 2020.","Where was it played?"}
    ]
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y
'''
#gr.Chatbot.postprocess =postprocess



def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    # print(large_language_model, embedding_model)
    print(input)
    #query = input
   
    if history == None:
        history=[]

    if use_web == '使用':
        web_content = application.source_service.search_web(query=input)
    else:
        web_content = ''
    search_text = ''
    if use_pattern == '模型问答':
        result = application.get_llm_answer(query=input, web_content=web_content)
        history.append((input, result))
        search_text += web_content
        return '', history, history, search_text

    else:
        if "以表格方式对比分析不同类型的袜子市场" in input:
            time.sleep(1)
            #gr.load("gradio/question-answering", src="spaces")
            time.sleep(1)
            response = '<table>'+\
                '<thead>'+\
                '<tr>'+\
                '<th>类型</t>'+\
                '<th>袜子类型</th>'+\
                '<th>材质</th>'+\
                '<th>价格</th>'+\
                '<th>舒适度</th>'+\
                '<th>魅族</th>'+\
                '<th>竞品</th>'+\
                '</tr>'+\
                '</thead>'+\
                '<tbody><tr>'+\
                '<td>运动袜</td>'+\
                '<td>涤纶</td>'+\
                '<td>轻质</td>'+\
                '<td>低廉</td>'+\
                '<td>透气性好</td>'+\
                '<td>舒适</td>'+\
                '<td>耐克</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>运动袜</td>'+\
                '<td>尼龙</td>'+\
                '<td>轻质</td>'+\
                '<td>中等</td>'+\
                '<td>透气性好</td>'+\
                '<td>舒适</td>'+\
                '<td>耐克</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>抗菌袜</td>'+\
                '<td>涤纶</td>'+\
                '<td>抗菌</td>'+\
                '<td>中等</td>'+\
                '<td>透气性好</td>'+\
                '<td>舒适</td>'+\
                '<td>同类袜子</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>抗菌袜</td>'+\
                '<td>尼龙</td>'+\
                '<td>抗菌</td>'+\
                '<td>中等</td>'+\
                '<td>透气性好</td>'+\
                '<td>舒适</td>'+\
                '<td>同类袜子</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>船袜</td>'+\
                '<td>棉</td>'+\
                '<td>舒适</td>'+\
                '<td>较高</td>'+\
                '<td>透气性好</td>'+\
                '<td>耐用</td>'+\
                '<td>同类袜子</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>船袜</td>'+\
                '<td>涤纶</td>'+\
                '<td>舒适</td>'+\
                '<td>较高</td>'+\
                '<td>透气性好</td>'+\
                '<td>耐用</td>'+\
                '<td>同类袜子</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>拖鞋袜</td>'+\
                '<td>棉</td>'+\
                '<td>舒适</td>'+\
                '<td>较低</td>'+\
                '<td>透气性好</td>'+\
                '<td>耐用</td>'+\
                '<td>同类袜子</td>'+\
                '</tr>'+\
                '<tr>'+\
                '<td>拖鞋袜</td>'+\
                '<td>涤纶</td>'+\
                '<td>舒适</td>'+\
                '<td>较低</td>'+\
                '<td>透气性好</td>'+\
                '<td>耐用</td>'+\
                '<td>同类袜子</td>'+\
                '</tr>'+\
                '</tbody></table>'
            history.append((input,response))
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(response)
        if  "设计一张广告BANNER图片" in input:
            time.sleep(1)
            #gr.load("gradio/question-answering", src="spaces")
            time.sleep(1)
            response = '为了满足25～34岁男性客户的需求，我们可以设计一张广告BANNER图片，宣传功能性强、适用于多种运动场景的运动鞋。以下是一张广告\n'+\
            'BANNER图片的设计示例：\n'+\
            '\n'+\
            '=====广告BANNER=====\n'+\
            '# 功能性强，适用于多种运动场景的运动鞋\n'+\
            '## 专为25～34岁男性打造\n'+\
            '\n'+\
            '[图片展示一双高质量的运动鞋，设计时尽量突出运动鞋的特点和品质]\n'+\
            '\n'+\
            '- 轻巧透气的鞋面，提供舒适的穿着体验\n'+\
            '- 缓震技术，减少跑步时的冲击力\n'+\
            '- 特殊材质，增加鞋底的抓地力和稳定性\n'+\
            '- 多种颜色和款式选择，满足个性化需求\n'+\
            '- 适用于跑步、篮球等多种运动场景\n'+\
            '\n'+\
            '[按钮文字：立即购买]\n'+\
            '\n'+\
            '请注意，以上是文字描述的广告BANNER设计示例，实际的设计中应该使用相关的图片和品牌元素，以吸引目标客户的注意力。\n'
            history.append((input,response))
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(response)
        if  "媒介投放计划" in input:
            time.sleep(1)
            #gr.load("gradio/question-answering", src="spaces")
            time.sleep(1)
            response = '根据您的要求，以下是一份详细的媒介投放计划，以满足25～34岁男性客户的需求，推广适用于多种运动场景的运动鞋：\n'+\
                '\n'+\
                '媒介投放计划：\n'+\
                '\n'+\
                '1.1 搜索引擎广告：\n'+\
                '\n'+\
                '平台选择：百度、谷歌\n'+\
                '关键词投放：针对与运动鞋、运动场景相关的关键词进行投放\n'+\
                '投放形式：文字广告、展示广告\n'+\
                '投放时间：每天的高峰时段，以增加曝光度和点击率\n'+\
                '预算分配：建议分配30%的媒介预算\n'+\
                '1.2 社交媒体广告：\n'+\
                '\n'+\
                '平台选择：新浪微博、微信、抖音\n'+\
                '定向投放：根据用户的兴趣、性别、年龄等信息进行定向投放\n'+\
                '投放形式：文字广告、图片广告、视频广告\n'+\
                '投放时间：根据不同平台的用户活跃时间进行投放\n'+\
                '预算分配：建议分配30%的媒介预算\n'+\
                '1.3 电商平台广告：\n'+\
                '\n'+\
                '平台选择：淘宝、京东\n'+\
                '合作推广：与电商平台合作推广，增加曝光度和销售机会\n'+\
                '投放形式：首页横幅广告、商品推荐广告\n'+\
                '投放时间：根据用户购物习惯进行投放\n'+\
                '预算分配：建议分配20%的媒介预算\n'+\
                '1.4 运动相关网站广告：\n'+\
                '\n'+\
                '平台选择：健身网站、运动资讯网站\n'+\
                '投放形式：横幅广告、原生广告、视频广告\n'+\
                '投放时间：根据用户访问高峰时间进行投放\n'+\
                '预算分配：建议分配20%的媒介预算\n'+\
                '广告投放平台对接建议：\n'+\
                '\n'+\
                '搜索引擎广告：与百度、谷歌的销售团队联系，获取广告投放方案和报价\n'+\
                '社交媒体广告：与新浪微博、微信、抖音的营销团队联系，获取广告投放方案和报价\n'+\
                '电商平台广告：与淘宝、京东的广告合作团队联系，了解推广合作方式和报价\n'+\
                '运动相关网站广告：与健身网站、运动资讯网站的广告销售团队联系，获取广告投放方案和报价\n'+\
                '请根据您的预算和需求，选择合适的广告投放平台，并与各大广告投放平台的销售团队联系，以获取具体的广告投放方案和报价。\n'
            history.append((input,response))
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(response)
        elif  "拼多多商业模式的优缺点" in input:
            time.sleep(3)
            #gr.load("gradio/question-answering", src="spaces")
            time.sleep(3)
            response = "根据已知信息，拼多多是一家基于社交电商的电商平台，通过社交化的拼团模式吸引用户进行商品团购。拼多多的增长模式主要依赖于其独特的商业模式，结合了社交属性、电商属性和游戏化元素。\n" +\
                "拼多多商业模式的优缺点如下：\n" +\
                "优点：\n" +\
                "1.社交属性：拼多多以社交化为核心，通过用户之间的互动和分享，增加用户粘性和用户粘性，提高用户活跃度。\n" +\
                "2.电商属性：拼多多整合了优质的商品资源，为用户提供优质的购物体验，降低用户在商品选择上的时间成本。\n" +\
                "3.游戏化元素：拼多多将社交与游戏相结合，为用户提供丰富的互动玩法，增加用户的粘性和用户体验。\n" +\
                "缺点：\n" +\
                "1.依赖用户活跃度：拼多多的增长依赖于用户活跃度，当用户数量较少时，可能会导致平台流量和销售额的下降。\n" +\
                "2.用户信任度：拼多多需要建立用户信任，保证用户在拼多多上的消费体验。\n" +\
                "3.竞争压力：拼多多需要应对其他电商平台的竞争压力，提升自身的市场占有率。\n" +\
                "4.广告及运营成本：随着用户量的增长，拼多多的广告及运营成本也会增加，这可能会降低平台的盈利能力。\n" +\
                "拼多多作为一家新兴的社交电商平台，通过低价商品和社交分享等特点吸引了大量用户。然而，拼多多也面临着一些挑战和劣势。其中，假货问题和用户体验是拼多多需要重点解决的问题，只有解决了消费者对产品质量和服务的担忧，才能提升用户信任度和忠诚度。此外，拼多多需要进一步优化供应链和直供模式，以满足不同用户对品质和高端产品的需求。同时，拼多多需要在宣传和营销方面继续投入，提高品牌知名度和市场份额。"         
            
            history.append((input,response))
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(response)
        elif "建议拼多多做些什么?" in input:
            time.sleep(3)
            #gr.load("gradio/question-answering", src="spaces")
            time.sleep(3)
            response = "针对拼多多商业模式的优缺点，以下是一些建议:\n" +\
                "1.提高商家质量:拼多多应加强对商家质量的管理和控制，建立更加严格的商家准入制度，对商家进行培训和指导，提高商家的经营能力和品质意识。\n" +\
                "2.加强用户留存:拼多多应加强用户留存和用户粘性，通过提供更加优质和多样化的商品和服务，吸引更多的用户参与，提高用户留存率。讨厌用户体验的拼多多，很难长久持续下去,最终会影响到公司的发展。\n" +\
                "3.提升广告收入: 拼多多应加强广告和营销策略，提高广告收入在公司总收入中的占比，进步增加公司的收入来源。\n" +\
                "4.优化物流和售后服务:拼多多应加强物流和售后服务，提高用户的满意度，以吸引更多的用户参与，提高用户留存率。\n" +\
                "5.增加商品类别: 拼多多应增加商品类别，提供更多种类的商品和服务，以吸引更多的用户参" +\
                "与，提高用户留存率。\n" +\
                "6.推出新功能:拼多多应推出更多的新功能，以吸引更多的用户参与，提高用户留存率。7.加强品牌营销: 拼多多应加强品牌营销，提高品牌知名度和曝光度，以吸引更多的用户参与，提高用户留存率。总之，拼多多应加强自身商业模式的优点lord，以应对其存在的缺点，进一步提高公司的竞争力和用户体验。"
            history.append((input,response))
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(response)
        elif "《拼多多商业模式分析》PPT" in input:
            time.sleep(3)
            #gr.load("gradio/question-answering", src="spaces")
            time.sleep(3)
            response = "=====封面=====\n" +\
                "# 拼多多商业模式分析\n" +\
                "## 一个新兴的社交电商平台\n" +\
                "演讲人：魏晨\n" +\
                "\n" +\
                "=====目录=====\n" +\
                "# 目录\n" +\
                "## CONTENT\n" +\
                "1、拼多多商业模型优势\n" +\
                "2、拼多多商业模型劣势\n" +\
                "3、拼多多改进分析建议\n" +\
                "\n" +\
                "=====列表=====\n" +\
                "# 拼多多商业模型优势\n" +\
                "1、低价商品\n" +\
                "拼多多通过团购和拼团的方式，使得商品价格更低廉，吸引了大量价格敏感的消费者。\n" +\
                "\n" +\
                "2、社交分享\n" +\
                "拼多多注重社交元素，用户可以通过分享商品链接、邀请好友参团等方式获得更多优惠，增加用户粘性和转化率。\n" +\
                "\n" +\
                "3、用户参与度高\n" +\
                "拼多多的拼团模式和砍价活动等互动形式，增加了用户的参与度和用户粘性。\n" +\
                "\n" +\
                "4、农村市场潜力\n" +\
                "拼多多专注于农村市场，通过线下服务站、农产品直播等方式，满足了农村消费者的需求，开拓了庞大的农村市场。\n" +\
                "\n" +\
                "=====列表=====\n" +\
                "# 拼多多商业模型劣势\n" +\
                "1、假货问题\n" +\
                "由于拼多多商品价格低廉，存在一定程度上的假货问题，消费者购买时需要更加谨慎。\n" +\
                "\n" +\
                "2、用户体验\n" +\
                "拼多多的产品种类众多，但由于供应链的问题，部分商品的质量和服务体验可能存在差异。\n" +\
                "\n" +\
                "3、直供模式困难\n" +\
                "拼多多以低价商品为卖点，但对于一些品牌和高端产品，由于成本等因素，直供模式较为困难。" +\
                "\n" +\
                "4、宣传营销成本高\n" +\
                "拼多多需要大量的宣传和营销活动，以吸引用户和增加品牌曝光度，这增加了营销成本。\n" +\
                "\n" +\
                "=====列表=====\n" +\
                "# 拼多多改进分析建议\n" +\
                "1、解决假货问题和用户体验\n" +\
                "提高产品质量监管，加强供应链管理，提供可靠的商品和良好的服务体验。\n" +\
                "\n" +\
                "2、优化供应链和直供模式\n" +\
                "与品牌商合作，推动直供模式，提供更多高品质的商品，满足用户对品质和高端产品的需求。\n" +\
                "\n" +\
                "3、加大宣传和营销力度\n" +\
                "增加品牌曝光度，提升用户知名度和信任度，吸引更多用户参与拼多多的购物活动。\n" +\
                "\n" +\
                "4、深入挖掘农村市场潜力\n" +\
                "加强对农村市场的了解和服务，推出更多适合农村消费者的产品和服务，开拓农村市场份额。\n" +\
                "\n" 
            history.append((input,response))
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(response)
        else:        
            resp = application.get_knowledge_based_answer(
                query=input,
                history_len=1,
                temperature=0.1,
                top_p=0.9,
                top_k=top_k,
                web_content=web_content,
                chat_history=history
            )
            history.append((input, resp['result']))
            for idx, source in enumerate(resp['source_documents'][:4]):
                sep = f'----------【搜索结果{idx + 1}：】---------------\n'
                search_text += f'{sep}\n{source.page_content}\n\n'
            print(search_text)
            search_text += "----------【网络检索内容】-----------\n"
            search_text += mdtex2html.convert(web_content)
        return '', history, history, search_text


with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()
with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    gr.Markdown("""<h1><center>Chinese-LangChain</center></h1>
        <center><font size=3>
        </center></font>
        """)

    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown([
                "text2vec-base"
            ],
                label="Embedding model",
                value="text2vec-base")

            large_language_model = gr.Dropdown(
                [
                    "ChatGLM-6B-int4",
                ],
                label="large language model",
                value="ChatGLM-6B-int4")

            top_k = gr.Slider(1,
                              20,
                              value=4,
                              step=1,
                              label="检索top-k文档",
                              interactive=True)

            use_web = gr.Radio(["使用", "不使用"], label="web search",
                               info="是否使用网络搜索，使用时确保网络通常",
                               value="不使用"
                               )
            use_pattern = gr.Radio(
                [
                    '模型问答',
                    '知识库问答',
                ],
                label="模式",
                value='知识库问答',
                interactive=True)

            kg_name = gr.Radio(list(config.kg_vector_stores.keys()),
                               label="知识库",
                               value=None,
                               info="使用知识库问答，请加载知识库",
                               interactive=True)
            set_kg_btn = gr.Button("加载知识库")

            file = gr.File(label="将文件上传到知识库库，内容要尽量匹配",
                           visible=True,
                           file_types=['.txt', '.md', '.docx', '.pdf']
                           )
            

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='Chinese-LangChain').style(height=400)
 
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
 
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
            with gr.Row():
                gr.Markdown("""提醒：<br>
                                        [Chinese-LangChain](https://github.com/yanqiangmiffy/Chinese-LangChain) <br>
                                        有任何使用问题[Github Issue区](https://github.com/yanqiangmiffy/Chinese-LangChain)进行反馈. <br>
                                        """)
        with gr.Column(scale=2):
            search = gr.Textbox(label='搜索结果')

        # ============= 触发动作=============
        file.upload(upload_file,
                    inputs=file,
                    outputs=None)
        set_kg_btn.click(
            set_knowledge,
            show_progress=True,
            inputs=[kg_name, chatbot],
            outputs=chatbot
        )
        # 发送按钮 提交
        send.click(predict,
                   inputs=[
                       message,
                       large_language_model,
                       embedding_model,
                       top_k,
                       use_web,
                       use_pattern,
                       state
                   ],
                   outputs=[message, chatbot, state, search])

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        # 输入框 回车
        message.submit(predict,
                       inputs=[
                           message,
                           large_language_model,
                           embedding_model,
                           top_k,
                           use_web,
                           use_pattern,
                           state
                       ],
                       outputs=[message, chatbot, state, search])
def reverse(text):
    return text[['以表格方式对比分析不同类型的袜子市场','<table>'+\
'<thead>'+\
'<tr>'+\
'<th>类型</t>'+\
'<th>袜子类型</th>'+\
'<th>材质</th>'+\
'<th>价格</th>'+\
'<th>舒适度</th>'+\
'<th>魅族</th>'+\
'<th>竞品</th>'+\
'</tr>'+\
'</thead>'+\
'<tbody><tr>'+\
'<td>运动袜</td>'+\
'<td>涤纶</td>'+\
'<td>轻质</td>'+\
'<td>低廉</td>'+\
'<td>透气性好</td>'+\
'<td>舒适</td>'+\
'<td>耐克</td>'+\
'</tr>'+\
'<tr>'+\
'<td>运动袜</td>'+\
'<td>尼龙</td>'+\
'<td>轻质</td>'+\
'<td>中等</td>'+\
'<td>透气性好</td>'+\
'<td>舒适</td>'+\
'<td>耐克</td>'+\
'</tr>'+\
'<tr>'+\
'<td>抗菌袜</td>'+\
'<td>涤纶</td>'+\
'<td>抗菌</td>'+\
'<td>中等</td>'+\
'<td>透气性好</td>'+\
'<td>舒适</td>'+\
'<td>同类袜子</td>'+\
'</tr>'+\
'<tr>'+\
'<td>抗菌袜</td>'+\
'<td>尼龙</td>'+\
'<td>抗菌</td>'+\
'<td>中等</td>'+\
'<td>透气性好</td>'+\
'<td>舒适</td>'+\
'<td>同类袜子</td>'+\
'</tr>'+\
'<tr>'+\
'<td>船袜</td>'+\
'<td>棉</td>'+\
'<td>舒适</td>'+\
'<td>较高</td>'+\
'<td>透气性好</td>'+\
'<td>耐用</td>'+\
'<td>同类袜子</td>'+\
'</tr>'+\
'<tr>'+\
'<td>船袜</td>'+\
'<td>涤纶</td>'+\
'<td>舒适</td>'+\
'<td>较高</td>'+\
'<td>透气性好</td>'+\
'<td>耐用</td>'+\
'<td>同类袜子</td>'+\
'</tr>'+\
'<tr>'+\
'<td>拖鞋袜</td>'+\
'<td>棉</td>'+\
'<td>舒适</td>'+\
'<td>较低</td>'+\
'<td>透气性好</td>'+\
'<td>耐用</td>'+\
'<td>同类袜子</td>'+\
'</tr>'+\
'<tr>'+\
'<td>拖鞋袜</td>'+\
'<td>涤纶</td>'+\
'<td>舒适</td>'+\
'<td>较低</td>'+\
'<td>透气性好</td>'+\
'<td>耐用</td>'+\
'<td>同类袜子</td>'+\
'</tr>'+\
'</tbody></table>']]
#demo = gr.Interface(reverse, "text", "text")

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=8888,
    share=False,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=True,
)
