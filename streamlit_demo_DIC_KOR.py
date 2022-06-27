########################################
# KOREA SECUIRTY EXCHANGE FIRM NETWORK #
#  VISUALIZATION VIA STREAMLIT (DEMO)  #
########################################
# First Created: 2022-06-23            #
# Last Updated: 2022-06-27             #
# Written By: Hye Jin Lee              #
#            (hyejinlee@dm.snu.ac.kr)  #
########################################

# PRELIMINARIES
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from networkx.classes import function as gf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from datetime import datetime

import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from streamlit import caching

btins = ['ALL','제 조 업','협회 및 단체, 수리 및 기타 개인 서비스업','예술, 스포츠 및 여가관련 서비스업','교육 서비스업','사업시설 관리, 사업 지원 및 임대 서비스업','전문, 과학 및 기술 서비스업','부동산업','금융 및 보험업','정보통신업','숙박 및 음식점업','운수 및 창고업','도매 및 소매업','건 설 업','수도, 하수 및 폐기물 처리, 원료 재생업','전기, 가스, 증기 및 공기 조절 공급업','광 업','농업, 임업 및 어업']

# Initialize Sidebar Menus
st.write('''
### KOREA STOCK EXCHANGE Firm Network (DEMO)
'''
)


# Get input from market and quarter filters
st.sidebar.header('Network Filters')
choose_mkt = st.sidebar.selectbox('Market',['KOSPI','KOSDAQ'])
choose_yr = st.sidebar.selectbox('Year',[2019])
choose_qtr = st.sidebar.selectbox('Quarter',['Q1','Q2','Q3','Q4'])



#choose_run = st.sidebar.button('Run!')
#my_slot1 = st.sidebar.empty()
#my_slot2 = st.sidebar.empty()
#my_slot1.info("Clear cache")
#if my_slot2.button("Clear"):
#    my_slot1.error("Do you really, really, wanna do this?")
#    if my_slot2.button("Yes I'm ready to rumble"):
#        caching.clear_cache()

# Load Data
G = nx.read_gpickle("./data/KOR_{}_2019{}_ALL.gpickle".format(choose_mkt,choose_qtr))
weights = [item[2]['weight'] for item in list(G.edges(data=True))]

# Network size by market cap
mkcaps = [item[1].get('시가총액') for item in sorted(G.nodes(data=True), key=lambda t: t[1].get('시가총액', 1), reverse=True)]

# Node size by market cap
mkcap_sizes = [firm[1]*10**-10 for firm in list(G.nodes(data='시가총액'))]

# 한글 표시를 위한 폰트 지정
font_name = fm.FontProperties(fname="./data/NanumBarunGothic.ttf").get_name()
matplotlib.rc('font', family=font_name)

# Node Labels by mkt cap
top10s_isin = [item[0] for item in sorted(G.nodes(data=True), key=lambda t: t[1].get('시가총액', 1), reverse=True)[:10]]
corp_names = nx.get_node_attributes(G, '종목명')

labels = {}
for node in G.nodes():
    if node in top10s_isin:
        # set corp ID as the key and corp name as the value
        labels[node] = corp_names[node]

# Define colormap for coloring nodes by industry
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the .jet map

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N) # create the new map

bounds = np.linspace(0, 18, 18) # define the bins
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N) # normalize the bins

colors = [item[1].get('GICS_code') for item in G.nodes(data=True)] # Extract industries to match colors

with open('./data/ColorLegend.pkl', 'rb') as handle:
    ColorLegend = pickle.load(handle)

# Announce additional sidebar options
choose_nodecoloring = st.sidebar.radio('Choose node coloring option',['None','By industry'])

if choose_nodecoloring=="None":
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, k=0.13, seed=705)
    nc = nx.draw_networkx_nodes(G,pos,node_size=mkcap_sizes,
                                node_color='cyan',
                                linewidths=0.5,
                                alpha=0.6
                                )
    nc.set_edgecolor('r')
    nx.draw_networkx_labels(G,pos,labels,font_family=font_name,font_size=8,font_color='black')

    ec = nx.draw_networkx_edges(G, pos, alpha=0.2,
                                edge_color=weights,
                                width=0.5,
                                edge_cmap=plt.cm.gist_gray)
    plt.axis('off')
    plt.title('통합종목네트워크\n ({yr} {qtr})'.format(yr=str(choose_yr), qtr=choose_qtr))
    plt.text(-1.5,-1.5,
             '{n_nodes} nodes, {n_edges} edges\nNetwork density = {density}\nClustering coefficient = {coef}'.format(n_nodes=G.number_of_nodes(),
                                                                                                                     n_edges=G.number_of_edges(),
                                                                                                                     density=round(nx.density(G),5),
                                                                                                                     coef=round(nx.average_clustering(G),5)))

    st.pyplot(fig)

else:
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, k=0.13, seed=705)
    nc = nx.draw_networkx_nodes(G,pos,node_size=mkcap_sizes,
                                node_color=colors, cmap=cmap,
                                linewidths=0.5,
                                alpha=0.6
                                )
    nc.set_edgecolor('r')
    nx.draw_networkx_labels(G,pos,labels,font_family=font_name,font_size=8,font_color='black')

    ec = nx.draw_networkx_edges(G, pos, alpha=0.2,
                                edge_color=weights,
                                width=0.5,
                                edge_cmap=plt.cm.gist_gray)

    cbar = plt.colorbar(nc,cmap=cmap,ticks=[category[0] for category in sorted(ColorLegend.items(), key=lambda item: item[0])])
    cbar.set_ticklabels([category[1] for category in sorted(ColorLegend.items(), key=lambda item: item[0])])

    plt.axis('off')
    plt.title('통합종목네트워크\n ({yr} {qtr})'.format(yr=str(choose_yr), qtr=choose_qtr))
    plt.text(-1.5,-1.5,
             '{n_nodes} nodes, {n_edges} edges\nNetwork density = {density}\nClustering coefficient = {coef}'.format(n_nodes=G.number_of_nodes(),
                                                                                                                     n_edges=G.number_of_edges(),
                                                                                                                     density=round(nx.density(G),5),
                                                                                                                     coef=round(nx.average_clustering(G),5)))

    st.pyplot(fig)

# Report top 5 degrees, betweenness, closeness nodes
attributes = pd.read_csv('./data/KOR_attributes_simple.csv')
attributes['종목코드'] = attributes['종목코드'].apply(lambda x: str('0'*(6-len(str(x)))+str(x)))
attributes.rename(columns = {'btin':'DIC'}, inplace = True)

top5s_degs = [item[0] for item in sorted(G.nodes(data=True), key=lambda t: t[1].get('degrees', 1), reverse=True)[:5]]
new1 = pd.DataFrame()
for i in range(5):
    new1 = new1.append(pd.DataFrame(attributes[attributes['종목코드']==top5s_degs[i]][['종목코드','종목명','시가총액_label','DIC','SIC','GICS']]), ignore_index=True)

top5s_bets = [item[0] for item in sorted(G.nodes(data=True), key=lambda t: t[1].get('betweenness', 1), reverse=True)[:5]]
new2 = pd.DataFrame()
for i in range(5):
    new2 = new2.append(pd.DataFrame(attributes[attributes['종목코드']==top5s_bets[i]][['종목코드','종목명','시가총액_label','DIC','SIC','GICS']]), ignore_index=True)

top5s_cls = [item[0] for item in sorted(G.nodes(data=True), key=lambda t: t[1].get('closeness', 1), reverse=True)[:5]]
new3 = pd.DataFrame()
for i in range(5):
    new3 = new1.append(pd.DataFrame(attributes[attributes['종목코드']==top5s_cls[i]][['종목코드','종목명','시가총액_label','DIC','SIC','GICS']]), ignore_index=True)


# top-level filters
stock_filter = st.selectbox("Select the stock", pd.unique(new2["종목코드"]))

# creating a single-element container.
placeholder = st.empty()

with placeholder.container():

# create two columns for charts
    st.markdown("### Top 5 **Bridging** Stocks")
    st.dataframe(new2)

with st.expander("Stock Movement in the Market"):
    start_date = st.date_input('Start date', datetime(2021,1,1))

    today = datetime.today().strftime('%Y-%m-%d')

    end_date = st.date_input('End date', datetime(int(today.split('-')[0]),int(today.split('-')[1]),int(today.split('-')[2])))

    prices = fdr.DataReader(stock_filter, start_date.strftime('%Y-%m-%d'),end_date.strftime('%Y-%m-%d'))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=prices.index, y=prices['Close'], name="{} Closing price".format(attributes.loc[attributes['종목코드'] == stock_filter, '종목명'].item()), mode="lines"),
        secondary_y=True
    )

    fig.add_trace(
        go.Bar(x=prices.index, y=prices['Volume'], name="{} Trading volume".format(attributes.loc[attributes['종목코드'] == stock_filter, '종목명'].item())),
        secondary_y=False
    )

    fig.update_xaxes(title_text="Date")

    fig.add_trace(
        go.Scatter(x=prices.index, y=prices['Open'], name="{} Opening price".format(attributes.loc[attributes['종목코드'] == stock_filter, '종목명'].item()), mode="lines"),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=prices.index, y=prices['High'], name="{} Highest price".format(attributes.loc[attributes['종목코드'] == stock_filter, '종목명'].item()), mode="lines"),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=prices.index, y=prices['Low'], name="{} Lowest price".format(attributes.loc[attributes['종목코드'] == stock_filter, '종목명'].item()), mode="lines"),
        secondary_y=True
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="Shares", secondary_y=False)
    fig.update_yaxes(title_text="WON", secondary_y=True)

    st.plotly_chart(fig)



# Announce rest of the sidebar options

choose_numNodes = st.sidebar.slider('Number of nodes in the network (by descending order of market cap)',5,len(mkcaps))
choose_DICindustry = st.sidebar.multiselect('DIC Industry (Multi-select)',btins)

#st.write(choose_mkt)

#SG=G.subgraph([node for node, data in G.nodes(data=True) if data.get("시장구분")==choose_mkt])

#st.write(len(SG.nodes))





# Announce rest of the sidebar options
#
#choose_DICindustry = st.sidebar.multiselect('DIC Industry (Multi-select)',industrylist)
#choose_firm = st.sidebar.multiselect('Secuity (Multi-select)',firmlist)


#firmlist = ['ALL']
#firmlist = firmlist.append(node_attr['종목명'].tolist())
#firmlist.sort()

#industrylist  = ['ALL']
#industrylist.extend(node_attr['btin'].tolist())


#---

#- Last Updated on June , 2022 by Hye Jin Lee ([hyejinlee@dm.snu.ac.kr](mailto:hyejinlee@dm.snu.ac.kr))
#- 2022 Copyright &copy; [서울대학교 산업공학과 데이터마이닝연구실](https://www.dm.snu.ac.kr)
#
#---
#''')