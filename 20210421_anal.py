#author: ohshirotakahito
#since: 202021/04/09

import numpy as np
import nptdms
import matplotlib.pyplot as plt
import math
from nptdms import TdmsFile
from nptdms import tdms

#tdms_file = TdmsFile.read(r'"D:personal\python\MLUT_NTO\data\d.tdms")
path_load ="a.tdms"

#Tdmsファイル読み込み
tdms_file = nptdms.TdmsFile(path_load)

#TDMSのツリー構造の取得
#Tdmsファイルのツリー構造のGroups名取得
tdms_groups = tdms_file.groups()

tdms_groups[0]
#print(tdms_groups[0])

#Tdmsファイルのツリー構造のChannel名取得
channels0 =tdms_groups[0].channels()
#print(channels0)
channels1 =tdms_groups[3].channels()
#print(channels1)

#Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
Y_data=tdms_file['Data']['Ch1'].raw_data

#Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
Data_name = tdms_file['AR Table']['Filename'].raw_data
Device_name = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
Ex_ID = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
Distance = tdms_file['AR Table']['Distance (nm)'].raw_data
Ex_ID = tdms_file['AR Table']['Ex ID'].raw_data

Distance = Distance[0]
Ex_ID = Ex_ID[0]

target0 = ' D_'
F_name = Data_name[0]
idx0 = F_name.find(target0)
Sample_name = F_name[:idx0+len(target0)-3]

target1 = 'DV_'
Device_name = Device_name[0]
idx1 = Device_name.find(target1)
Gap_ID = Device_name[idx1+len(target1):]

target2 = '-G'
idx2 = Gap_ID.find(target2)
Device_ID = Gap_ID[:idx2+len(target2)-2]

target3 = ' A'
idx3 = Ex_ID.find(target3)
Date_ID = Ex_ID[:idx3+len(target3)-2]
PDate = Ex_ID[:idx3+len(target3)-1]

target4 = 'Pex'
idx4 = PDate.find(target4)
Machine_ID = PDate[:idx4+len(target4)-3]

#S_Table dataの再構成
SM_data=tdms_file['S Table']['S # [n]'].raw_data
SP_data=tdms_file['S Table']['S Peak Position [s]'].raw_data
SI_data=tdms_file['S Table']['Signal [pA]'].raw_data
SB_data=tdms_file['S Table']['Region BL [pA]'].raw_data
SD_data=tdms_file['S Table']['Region STD [pA]'].raw_data
SS_data=tdms_file['S Table']['Signal S [s]'].raw_data
SE_data=tdms_file['S Table']['Signal E (s)'].raw_data
ST_data=tdms_file['S Table']['S TL [ms]'].raw_data
SL_data=tdms_file['S Table']['S DL [s]'].raw_data

#S_Table dataの少数データ化
SM_data= [int(s) for s in SM_data]
SP_data= [float(s) for s in SP_data]
SI_data= [float(s) for s in SI_data]
SB_data= [float(s) for s in SB_data]
SD_data= [float(s) for s in SB_data]
SS_data= [float(s) for s in SS_data]
SE_data= [float(s) for s in SE_data]
ST_data= [float(s) for s in ST_data]
SL_data= [float(s) for s in SL_data]

# =============================================================================
# #S_Table dataの範囲絞り込み（テストのため）
# ds1=0
# dl1=1000
# 
# SM_data = SM_data[ds1:dl1]
# SP_data = SP_data[ds1:dl1]
# SI_data = SI_data[ds1:dl1]
# SB_data = SB_data[ds1:dl1]
# SD_data = SD_data[ds1:dl1]
# SS_data = SS_data[ds1:dl1]
# SE_data = SE_data[ds1:dl1]
# ST_data = ST_data[ds1:dl1]
# SL_data = SL_data[ds1:dl1]
# =============================================================================

#S dataのビューア範囲絞り込み（テストのため）
r=100

AX=[]

for m in range(len(SP_data)):
    XX=[]
    PX=[]
    EX=[]
    xm_data =[]
    y0_data =[]
    y1_data =[]
    y2_data =[]
    sm = SM_data[m]
    sp = SP_data[m]*10000
    ss = SS_data[m]*10000
    se = SE_data[m]*10000
    st = ST_data[m]*10
    si = SI_data[m]
    sb = SB_data[m]
    sd = SD_data[m]
    sl = SL_data[m]*10000
        
    sp= int(sp)
    ss= int(ss)
    se= int(se)
    st= int(st)
    
    sb=round(sb,6)
    sd=round(sd,6)
    sl=round(sl,6)
    
    #シグナル領域２の設定（シグナル領域の100%を採用するし，前後に配置する（st/2））
    mr=st/2
    mr=int(mr)
    #シグナル領域３の設定（シグナル領域の300%を採用するし，前後に配置する（st*3））
    me=st*3
    me=int(me)
    
    #EXの情報の記述
    print('+++++++++++++++++name++++++++++++++++++++++++')
    print('E1:Ex_ID:',Ex_ID)
    e1=Ex_ID
    EX.append(e1)
    print('E2:Gap_ID:',Gap_ID)
    e2=Gap_ID
    EX.append(e2)
    print('E3:Device_ID:',Device_ID)
    e3=Device_ID
    EX.append(e3)
    print('E4:PDate:',PDate)
    e4=PDate
    EX.append(e4)
    print('E5:Machine_ID:',Machine_ID)
    e5=Machine_ID
    EX.append(e5)
    print('E6:Distance(nm):',Distance)
    e6=Distance
    EX.append(e6)
    print('E7:Sample_name:',Sample_name)
    e7=Sample_name
    EX.append(e7)
    
    
    #シグナル領域2の抽出
    y0_data = Y_data[ss-1-mr:se+1+mr]
    s_max = np.amax(y0_data)
    #本当のピークを確認
    s_max_index = np.where(y0_data == s_max)
    xrmx= (s_max_index[0])[0]+ss-1-mr
    yrmx= Y_data[xrmx]
    
    #本当のシグナル開始を確認
    y1_data = Y_data[ss-1-mr:xrmx]
    #print(y1_data)
    if ss-1-mr==xrmx:
        s_min_1=xrmx
        s_min1_index = Y_data[xrmx]
        #print(s_min1_index)
        xrmn1= ss-1-mr
    else:
        s_min_1 = np.amin(y1_data)
        s_min1_index = np.where(y1_data == s_min_1)
        #print(s_min1_index)
        xrmn1= (s_min1_index[0])[0]+ss-1-mr
    yrmn1= Y_data[xrmn1]
    
    #本当のシグナル終了を確認
    y2_data = Y_data[xrmx:se+1+mr]
    s_min_2 = np.amin(y2_data)
    s_min2_index = np.where(y2_data == s_min_2)
    xrmn2= (s_min2_index[0])[0]+xrmx
    yrmn2= Y_data[xrmn2]
    
    #Sの情報の記述
    print('+++++++++++++++++peak++++++++++++++++++++++++')
    print('E8:signal_number(n):',int(sm))
    e8=int(sm)
    EX.append(e8)
    print('E9:signal_p_position:',sp/1000,'[s]')
    e9=sp/1000
    EX.append(e9)
    print('E10:signal_Tp_position',xrmx/10000,'[s]') 
    e10=xrmx/10000
    EX.append(e10)
    
    print('+++++++++++++++++start++++++++++++++++++++++++++')
    print('E11:signal_1st_region:Start',(ss-1)/10000,'[s]')
    e11=(ss-1)/10000
    EX.append(e11)
    print('E12:signal_2nd_region:Start',(ss-1-mr)/10000,'[s]')
    e12=(ss-1-mr)/10000
    EX.append(e12)
    print('E13:signal_3rd_region:Start',(ss-1-me)/10000,'[s]')
    e13=(ss-1-me)/10000
    EX.append(e13)
    print('E14:signal_Tp_region:Start',xrmn1/10000,'[s]') 
    e14=xrmn1/10000
    EX.append(e14)
    
    print('++++++++++++++++++end++++++++++++++++++++++++++')
    print('E15:signal_1st_region:End',(se+1)/10000,'[s]')
    e15=(se+1)/10000
    EX.append(e15)
    print('E16:signal_2nd_region:End',(se+1+mr)/10000,'[s]')
    e16=(se+1+mr)/10000
    EX.append(e16)
    print('E17:signal_3rd_region:End',(se+1+me)/10000,'[s]')
    e17=(se+1+me)/10000
    EX.append(e17)
    print('E18:signal_Tp_region:End',xrmn2/10000,'[s]')
    e18=xrmn2/10000
    EX.append(e18)
    
    print('+++++++++++++++++length+++++++++++++++++++++++++++')
    print('E19:signal_1st_region:lenth',(se-ss+2)/10,'[ms]')
    e19=(se-ss+2)/10
    EX.append(e19)
    print('E20:signal_2nd_region:length',(se-ss+2+2*mr)/10,'[ms]')
    e20=(se-ss+2+2*mr)/10
    EX.append(e20)
    print('E21:signal_3rd_region:length',(se-ss+2+2*me)/10,'[ms]')
    e21=(se-ss+2+2*me)/10
    EX.append(e21)
    print('E22:signal_Tp_region:Length',(xrmn2-xrmn1+1)/10,'[ms]')
    e22=(xrmn2-xrmn1+1)/10
    EX.append(e22)
    
    print('++++++++++++++++pre_region++++++++++++++++++++++++++')
    print('E23:signal_pre_region:Start',(ss-1-me)/10000,'[s]')
    e23=(ss-1-me)/10000
    EX.append(e23)
    print('E23:signal_pre_region:End',(ss-1)/10000,'[s]')
    e24=(ss-1)/10000
    EX.append(e24)
    print('E24:signal_pre_region:Length',mr/10,'[ms]')
    e25=mr/10
    EX.append(e25)
    
    print('++++++++++++++++post_region+++++++++++++++++++++++')
    print('E25:signal_post_region:Start',(se+1)/10000,'[s]')
    e25=(se+1)/10000
    EX.append(e25)
    print('E26:signal_post_region:End',(ss+1+me)/10000,'[s]')
    e26=(ss+1+me)/10000
    EX.append(e26)
    print('E27:signal_post_region:Length',me/10,'[ms]')
    e27=me/10
    EX.append(e27)
    
    
    #本当のBaselineを計算
    yyb = (yrmn1+yrmn2)/2
    yyb = round(yyb*1000,4)
    #本当のSignal Intensityを計算
    yyi = yrmx-(yrmn1+yrmn2)/2
    yyi = round(yyi*1000,4)
    s_upspd = (yrmx-yrmn1)*1000/(xrmx-ss+1)
    s_dnspd = (yrmx-yrmn2)*1000/(se+1-xrmx)
    s_upspd = round(s_upspd,6)
    s_dnspd = round(s_dnspd,6)
    
    print('________________________________________')
    print('P1:signal_intensity',si,'[pA]')
    p1=si#P1:signal_intensity[pA]
    PX.append(p1)
    print('P2:signal_t_intensity',yyi,'[pA]')
    p2=yyi#P2:signal_t_intensity[pA]
    PX.append(p2)
    print('P3:signal_baseline',sb,'[pA]')
    p3=sb#P3:signal_baseline[pA]
    PX.append(p3)
    print('P4:signal_t_baseline',yyb,'[pA]')
    p4=yyb#4:signal_t_baseline [pA]
    PX.append(p4)
    print('P5*signal_std',sd,'[pA]')
    p5=sd#P5*signal_std',sd,'[pA]'
    PX.append(p5)
    print('P6*signal_time',(se-ss+2+2*mr)/10,'[ms]')
    p6=(se-ss+2+2*mr)/10#P6*signal_time [ms]
    PX.append(p6)
    print('P7:signal_pretime',sl,'[ms]')
    p7=sl#P7:signal_pretime',sl,'[ms]'
    PX.append(p7)
    print('P8:signal_upspeed',s_upspd,'[pA/ms]')
    p8=s_upspd#P8:signal_upspeed',s_upspd,'[pA/ms]
    PX.append(p8)
    print('P9:signal_uptime',(xrmx-ss+1)/10,'[ms]')
    p9=(xrmx-ss+1)/10#P9:signal_uptime',(xrmx-ss+1)/10,'[ms]
    PX.append(p9)
    print('P10:signal_upcurrent',round((yrmx-yrmn1)*1000,6),'[pA]')
    p10=round((yrmx-yrmn1)*1000,6)#P10:signal_upcurrent',round((yrmx-yrmn1)*1000,6),'[pA]
    PX.append(p10)
    print('P11:signal_downspeed',s_dnspd,'[pA/ms]')
    p11=s_dnspd#P11:signal_downspeed',s_dnspd,'[pA/ms]
    PX.append(p11)
    print('P12:signal_downtime',(se+1-xrmx)/10,'[ms]')
    p12=(se+1-xrmx)/10#P12:signal_downtime',(se+1-xrmx)/10,'[ms]'
    PX.append(p12)
    print('P13:signal_downcurrent',round((yrmx-yrmn2)*1000,6),'[pA]')
    p13=round((yrmx-yrmn2)*1000,6)#P13:signal_downcurrent',round((yrmx-yrmn2)*1000,6),'[pA]
    PX.append(p13)
    
#シグナルPre領域の抽出（領域３－領域２の前半）
    #print(ss-1+me,ss-1-mr)
    pr_data = Y_data[ss-1-mr:ss-1+me]
    #print(pr_data)
    
    pr_average = np.average(pr_data)
    pr_average =round(pr_average*1000-yyb,5)
    pr_std = np.std(pr_data-yyb)
    pr_std =round(pr_std*1000-yyb,5)
    pr_max = np.amax(pr_data)*1000-yyb
    pr_min = np.amin(pr_data)*1000-yyb
    pr_max = round(pr_max,6)
    pr_min = round(pr_min,6)
    print('P14:pre_average',pr_average,'[pA]')
    p14=pr_average#P14:pre_average',pr_average,'[pA]
    PX.append(p14)
    print('P15:pre_std',pr_std,'[pA]')
    p15=pr_std#P15:pre_std',pr_std,'[pA]
    PX.append(p15)
    print('P16:pre_max',pr_max,'[pA]')
    p16=pr_max#P16:pre_max',pr_max,'[pA]
    PX.append(p16)
    print('P17:pre_min',pr_min,'[pA]')
    p17=pr_min#P17:pre_min',pr_min,'[pA]
    PX.append(p17)
    
    #シグナルPost領域2の抽出（領域３－領域２の前半）
    po_data = Y_data[se+1+mr:se+1+me]
    #print(se+1+mr,se+1+me)
    po_average = np.average(po_data)
    po_average =round(po_average*1000-yyb,5)
    po_std = np.std(po_data-yyb)
    po_std =round(po_std*1000,5)
    po_max = np.amax(po_data)*1000-yyb
    po_min = np.amin(po_data)*1000-yyb
    po_max = round(po_max,6)
    po_min = round(po_min,6)
    print('P18:post_average',po_average,'[pA]')
    p18=po_average#P18:post_average',po_average,'[pA]
    PX.append(p18)
    print('P19*post_std',po_std,'[pA]')
    p19=po_std#P19*post_std',po_std,'[pA]
    PX.append(p19)
    print('P20:post_max',po_max,'[pA]')
    p20=po_max#P20:post_max',po_max,'[pA]
    PX.append(p20)
    print('P21:post_min',po_min,'[pA]')
    p21=po_min#P21:post_min',po_min,'[pA]'
    PX.append(p21)
    
    
    #シグナル分割数
    sn=13
    e28=sn-1
    EX.append(e28)
    
    sz=[]
    th=0
    
    #シグナルベクトル作成：シグナル分割案１(np.array_split)
    #ssl=np.array_split(y0_data,sn)
    #for u in range(sn):
       #print('s_region',u,':',ssl[u])
    
    #シグナルベクトル作成：シグナル分割2(シグナル2領域)
    sru=len(y0_data)/sn
    y=[]
    for u in range(sn-1):
        x=sru*u
        x=round(x,5)
        mc=math.ceil(sru*u)
        mf=math.floor(sru*u)
        x1=mf
        if mc==len(y0_data):
            mc==len(y0_data)-1
            
        x2=mf+1
        mz=mc-mf
        #rint(x,mc,mf,mz)
        if mz<1:
            y1=y0_data[mf]
            y2=y0_data[mc]
            y=y1
        else:
            y1=round(y0_data[mf],5)
            y2=round(y0_data[mc-1],5)
            y=(y2-y1)/(x2-x1)*(x-x1)+y1
                    
        y=round(y*1000-yyb,5)
        y=round(y/yyi,5)
        y1=round(y1*1000-yyb,5)
        y2=round(y2*1000-yyb,5)
        x=round(x,5)
        PX.append(y)
        
        print('P',u+22,':','shape_factor',u,':',y)
    
    #プロット図作成
    if sp<r:
        ym_data = Y_data[0:sp+r]
    else:
        ym_data = Y_data[sp-r:sp+r]
    
    ym_n = len(ym_data)
    #print(ym_data)
    for m in range(ym_n):
        xm = m+sp-r
        xm_data.append(xm)
    xm_n = len(xm_data)
    
    XX=EX+PX
    AX.append(XX)
    
    #print(xm_data)
    plt.scatter(xrmx,yrmx,c="black")
    plt.scatter(sp,Y_data[sp])
    plt.scatter(ss-1,Y_data[ss-1],c="yellow")
    plt.scatter(se+1,Y_data[se+1],c="yellow")
    plt.scatter(xrmn1,yrmn1,c="red")
    plt.scatter(xrmn2,yrmn2,c="red")
    plt.scatter(ss-1-mr,Y_data[ss-1-mr],c="green")
    plt.scatter(se+1+mr,Y_data[se+1+mr],c="green")
    plt.scatter(ss-1-me,Y_data[ss-1-me],c="purple")
    plt.scatter(se+1+me,Y_data[se+me+1],c="purple")
    plt.plot(xm_data, ym_data)
    plt.show()


AX= [str(s) for s in AX]
save_path = (str(e7)+'.npy')
np.save(save_path, AX)
