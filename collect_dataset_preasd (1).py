#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from verticapy import vDataFrame
#from verticapy import vertica_conn
from verticapy import vdf_from_relation
import verticapy
import json


# In[2]:


import vertica_python

import datetime
import re


# In[3]:


with open('config.json', 'r') as f:
    conn_info = json.load(f)


# In[4]:


#почистим все сессии

with vertica_python.connect(**conn_info) as conn:
    sql = """
    SELECT close_user_sessions('abrut-brulyako');
    """

    cur = conn.cursor()
    cur.execute(sql)
    print(cur.fetchall())


# In[5]:


# test schema IVOLGA

with vertica_python.connect(**conn_info) as conn:
    sql = """
    SELECT DISTINCT table_name, table_type FROM all_tables
          WHERE SCHEMA_NAME = 'IVOLGA';
    """

    cur = conn.cursor()
    
    cur.execute(sql)
    print(cur.fetchall())


# In[6]:


verticapy.new_connection({
                   'host': conn_info['host'], 
                   'port': conn_info['port'], 
                   'database': conn_info['database'], 
                   'password': conn_info['password'], 
                   'user': conn_info['user'],
                  },
                   name = "MyVerticaConnection")


# In[7]:


verticapy.connect("MyVerticaConnection")


# In[8]:


verticapy.set_option("temp_schema", "IVOLGA")


# In[9]:


start_time = datetime.datetime.now()
duration = False
start_time


# In[10]:


end_date = datetime.datetime.now() - datetime.timedelta(2)
end_date_ = end_date.strftime('%Y-%m-%d')


# In[11]:


end_date_


# In[12]:


def collect_feature(sql, f_name):
    """
    select feature using sql and add this as column to table ivolga.preasd_t3:
    1. put table [user_id, f_name] into ivolga.preasd_t2
    2. set ivolga.preasd_t3 as join ivolga.preasd_t1 and ivolga.preasd_t2 on user_id
    3. set ivolga.preasd_t1 as current ivolga.preasd_t3
    """
        
    assert f_name
    assert sql
    assert type(sql) == str
    assert type(f_name) == str
    
    with vertica_python.connect(**conn_info) as conn:
        cur = conn.cursor()
        cur.execute('drop table if exists ivolga.preasd_t2')

        cur.execute(f'create table ivolga.preasd_t2 (user_id int, {f_name} float)')

        sql = f"""
            insert into ivolga.preasd_t2 (user_id, {f_name})
            {sql}
            """

        cur.execute(sql)

        cur.execute('commit;')
    
    with vertica_python.connect(**conn_info) as conn:
        cur = conn.cursor()
        cur.execute('drop table if exists ivolga.preasd_t3')

        cur.execute(f"""create table ivolga.preasd_t3 as select
                    t1.*, ISNULL(t2.{f_name},0) as {f_name} from 
                    ivolga.preasd_t1 t1 left join
                    ivolga.preasd_t2 t2 on t1.user_id = t2.user_id
                    """)

        cur.execute('drop table if exists ivolga.preasd_t1')

        cur.execute("""create table ivolga.preasd_t1 as select
                    * from ivolga.preasd_t3
                    """)

        cur.execute('commit;')    
        
    return True


# In[13]:


def transliterate(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r'[^a-zа-яё0-9 ]', '', s)

    cyr = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    en = 'abvgdeejziyklmnoprstufhccsh_y_euj'
    translit = dict(zip(cyr, en))
    
    res = ''
    for char in s:
        if char in translit.keys():
            res += translit[char]
        else:
            res += char
    res = re.sub(r'[ ]+', '_', res)
    return res
    


# In[14]:


days = 30
sql_active_sellers = f"""
        select DISTINCT cu.User_id
        from DMA.current_user cu
        right join DMA.current_item ci ON cu.User_id = ci.User_id
        right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
        WHERE ci.Microcat_id IN
        (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
        and subcategory_slug = 'vakansii' )
        AND cpe.event_time >= '{end_date}'::date - {days}
        and cpe.event_time <= '{end_date}'::date
        and cpe.is_revenue = true 
        and cu.user_id not in (select user_id from dma.current_user where isTest)
        and ci.isDead = false
        """


# In[15]:


# active sellers


# In[16]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    
    cur.execute('drop table if exists ivolga.preasd_t1')

    cur.execute('create table ivolga.preasd_t1 (user_id int, external_id int)')

    cur.execute(f"""insert into ivolga.preasd_t1  
        select DISTINCT cu.User_id, cu.External_id
        from DMA.current_user cu
        right join DMA.current_item ci ON cu.User_id = ci.User_id
        right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
        WHERE ci.Microcat_id IN
        (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
        and subcategory_slug = 'vakansii' )
        AND cpe.event_time >= '{end_date}'::date - {days}
        and cpe.event_time <= '{end_date}'::date
        and cpe.is_revenue = true 
        and cu.user_id not in (select user_id from dma.current_user where isTest)
        and ci.isDead = false
    """)

    cur.execute('commit;')


# In[17]:


# is ASD

sql = f"""
        select distinct user_id, 1
        from dma.am_client_day 
        where personal_manager_team ilike '%работа%' 
        and event_date = '{end_date_}'::date
        """
f_name = 'is_asd'

collect_feature(sql, f_name)


# In[18]:


# was ASD

sql = f"""
        select distinct user_id, 1
        from dma.am_client_day
        where personal_manager_team ilike '%работа%'
        """
f_name = 'was_asd'

collect_feature(sql, f_name)


# In[19]:


# others ASD

sql = f"""
        select distinct user_id, 1
        from dma.am_client_day 
        where personal_manager_team not ilike '%работа%' 
        and event_date >= '{end_date}'::date
        and user_id in ( 
                    {sql_active_sellers}
            )
        """
f_name = 'asd_others'

collect_feature(sql, f_name)


# In[20]:


# выручка за 90 дн в по всем вертикалям (берем период 90 дней от даты выгрузки)

sql = f"""
        select vas_user_id as user_id, sum(cp.amount_net)::float as revenue
        from dma.current_payment cp
        where vas_user_id in ( 
                {sql_active_sellers}
            )
            and is_revenue
            and item_is_dead = false
            and cp.event_time >= '{end_date}'::date-90
            and cp.event_time <= '{end_date}'::date
        group by user_id
        """
f_name = 'all_revenue_90'

collect_feature(sql, f_name)


# In[21]:


# суммарная выручка по всем вертикалям за последние 30 дн

sql = f"""
        select vas_user_id as user_id, sum(cp.amount_net)::float as revenue
        from dma.current_payment cp
        where vas_user_id in ( 
                {sql_active_sellers}
            )
             and is_revenue
            and item_is_dead = false
            and cp.event_time >= '{end_date}'::date-30
            and cp.event_time <= '{end_date}'::date
        group by user_id
        """
f_name = 'all_revenue_30'

collect_feature(sql, f_name)


# In[22]:


# количество активаций вакансий (объявлений в работе) по периодe 0-30

sql = f"""
    select 
        ci.User_id, 
        count(ci.item_id)::float
    from 
        dma.item_activations ia
        join DMA.current_item ci on ia.item_id = ci.Item_id
    where 
        activation_time::date >= '{end_date}'::date-30
        and activation_time::date <= '{end_date}'::date
        and activation_type in ('Package', 'Package_subs_afterpaid', 'Package_subs_prepaid', 'Single', 'Single_subs_afterpaid') -- платные активации
        and ci.user_id in ( 
                {sql_active_sellers}
            )
        and ci.Microcat_id IN (SELECT distinct  Microcat_id FROM DMA.current_microcategories where cat_id = 250006 )
    group by 
        ci.User_id
    """

f_name = 'activations_30'

collect_feature(sql, f_name)


# In[23]:


# количество активаций вакансий (объявлений в работе) по периоle 30-60

sql = f"""
    select 
        ci.User_id, 
        count(ci.item_id)::float
    from 
        dma.item_activations ia
        join DMA.current_item ci on ia.item_id = ci.Item_id
    where 
        activation_time::date >= '{end_date}'::date - 60
        and activation_time::date < '{end_date}'::date - 30
        and activation_type in ('Package', 'Package_subs_afterpaid', 'Package_subs_prepaid', 'Single', 'Single_subs_afterpaid') -- платные активации
        and ci.user_id in ( 
                {sql_active_sellers}
            )
        and ci.Microcat_id IN (SELECT distinct  Microcat_id FROM DMA.current_microcategories where cat_id = 250006 )
    group by 
        ci.User_id
    """

f_name = 'activations_30_60'

collect_feature(sql, f_name)


# In[24]:


# количество активаций вакансий (объявлений в работе) по периоle 60-90

sql = f"""
    select 
        ci.User_id, 
        count(ci.item_id)::float
    from 
        dma.item_activations ia
        join DMA.current_item ci on ia.item_id = ci.Item_id
    where 
        activation_time::date >= '{end_date}'::date - 90
        and activation_time::date < '{end_date}'::date - 60
        and activation_type in ('Package', 'Package_subs_afterpaid', 'Package_subs_prepaid', 'Single', 'Single_subs_afterpaid') -- платные активации
        and ci.user_id in ( 
                {sql_active_sellers}
            )
        and ci.Microcat_id IN (SELECT distinct  Microcat_id FROM DMA.current_microcategories where cat_id = 250006 )
    group by 
        ci.User_id
            """

f_name = 'activations_60_90'

collect_feature(sql, f_name)


# In[25]:


# количество активных объявлений в других вертикалях (не учитывать вакансии и резюме) на дату выгрузки (это по таблице dma.item_day)

sql = f"""
    select 
        ci.User_id, 
        count(id.item_id)::float
    from 
        DMA.item_day id
        join DMA.current_item ci on id.item_id = ci.Item_id
    where 
        id.event_date::date = '{end_date}'
        and ci.User_id in ( 
                {sql_active_sellers}
            )
        and ci.Microcat_id not IN (SELECT distinct  Microcat_id FROM DMA.current_microcategories where cat_id in (250004, 250006) )
    group by ci.User_id
"""

f_name = 'other_vertical_items_end_date'

collect_feature(sql, f_name)


# In[26]:


# объем продаж услуг юнита jobs за период 180 по селлеру

sql = f"""
    select vas_user_id as user_id, sum(cp.amount_net)::float as revenue
    from dma.current_payment cp
    where vas_user_id in ( 
            {sql_active_sellers}
        )
         and is_revenue
         and (
             (item_category_id = 250006)  
             or (item_category_id = 250004 and transaction_type = 'paid_contact')  
             or ( transaction_subtype ilike '%jobs%')
              )
        and item_is_dead = false
        and cp.event_time >= '{end_date}'::date-180
        and cp.event_time <= '{end_date}'::date
    and cp.vas_user_id not in (select cu.user_id from dma.current_user as cu where isTest)
    group by user_id
"""

f_name = 'revenue_180'

collect_feature(sql, f_name)


# In[27]:


# Количество объявлений юнита jobs за период по селлеру

sql = f"""
    select  cu.User_id, count(distinct ci.Item_id)::float as count_items
    from DMA.current_user cu
    right join DMA.current_item ci ON cu.User_id = ci.User_id
    right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
    WHERE ci.Microcat_id IN
    (SELECT distinct  Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
    and subcategory_slug = 'vakansii' )
    AND cpe.event_time >= '{end_date}'::date-180
    and cpe.event_time <= '{end_date}'::date
    and cu.user_id in ( 
            {sql_active_sellers}
        )
    and ci.isDead = false
    group by cu.User_id
    """

f_name = 'count_items_180'

collect_feature(sql, f_name)


# In[28]:


# Количество опллат объявлений юнитов jobs за период по селлеру

sql = f"""
    select  cu.User_id, count(distinct cpe.external_operation_id)::float as count_payments  
    from DMA.current_user cu
    right join DMA.current_item ci ON cu.User_id = ci.User_id
    right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
    WHERE ci.Microcat_id IN
    (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
    and subcategory_slug = 'vakansii' )
    AND cpe.event_time >= '{end_date}'::date-180
    and cpe.event_time <= '{end_date}'::date
    and cpe.is_revenue = true 
    and cu.user_id not in (select user_id from dma.current_user where isTest)
    and cu.user_id in ( 
            {sql_active_sellers}
        )
    and ci.isDead = false
    group by cu.User_id
    order by count_payments desc
    """

f_name = 'count_payments_180'

collect_feature(sql, f_name)


# In[29]:


# количество приобретения платных услуг по юниту jobs у селлера за период:

services = ['Highlight','XL','max single','mid single','x10_1','x10_7','x2_1','x2_7','x5_1','x5_7']

for service in services:
    sql = f"""
        select  cu.User_id, count(distinct cpe.external_operation_id)::float as cnt  
        from DMA.current_user cu
        right join DMA.current_item ci ON cu.User_id = ci.User_id
        right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
        WHERE ci.Microcat_id IN 
        (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
        and subcategory_slug = 'vakansii' )
        and cpe.transaction_subtype = '{service}'
        AND cpe.event_time >= '{end_date}'::date-180
        and cpe.event_time <= '{end_date}'::date
        and cpe.is_revenue = true 
        and cu.user_id not in (select user_id from dma.current_user where isTest)
        and cu.user_id in ( 
            {sql_active_sellers}
        )
        and ci.isDead = false
        group by cu.User_id
        order by cnt desc
        """
    
    f_name = service.replace(' ', '_')
    print(f_name, collect_feature(sql, f_name))


# In[30]:


# количество приобретения платных услуг по юниту jobs у селлера за период:

services = ['tariff ext Job', 'tariff ext Job ASD', 'tariff max Job', 'tariff max Job ASD', 
            'tariff lf package', 'single view', 'view from package'] 

for service in services:
    sql = f"""
        select
            cp.vas_user_id, count(distinct cp.VASFact_id)::float as cnt
        from dma.current_payment cp
        where True
              and cp.is_revenue
              and (not user_is_test or user_is_test is null)
              and (not item_is_dead or item_is_dead is null)
              and cp.event_time >= '{end_date}'::date-180
              and cp.event_time <= '{end_date}'::date
              and (
                  (item_category_id = 250006)       
                  or (item_category_id = 250004 and transaction_type = 'paid contact')           
                  or ( transaction_subtype ilike '%job%')
                   )
            and transaction_subtype = '{service}' 
            and cp.vas_user_id in ( 
                {sql_active_sellers}
            )
        group by 1
        order by cnt desc
        """
    
    f_name = service.replace(' ', '_')
    print(f_name, collect_feature(sql, f_name))


# In[31]:


# все субкатегории вакансий селлеров

with vertica_python.connect(**conn_info) as conn:

    sql = f"""
    select  distinct ci.Param1
    from DMA.current_user cu
    right join DMA.current_item ci ON cu.User_id = ci.User_id
    right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
    WHERE ci.Microcat_id IN
    (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
    and subcategory_slug = 'vakansii' )
    AND cpe.event_time >= '{end_date}'::date-180
    and cpe.event_time <= '{end_date}'::date
    and cpe.is_revenue = true 
    and cu.user_id not in (select user_id from dma.current_user where isTest)
    and cu.user_id in ( 
            {sql_active_sellers}
        )
    and ci.isDead = false
    """

    cur = conn.cursor()
    cur.execute(sql)
    Param1 = cur.fetchall()


# In[32]:


# количество вакансий данного вида по селлерам

for item in Param1:
    sql = f"""
        select  cu.User_id, count(ci.Param1)::float as cnt
        from DMA.current_user cu
        right join DMA.current_item ci ON cu.User_id = ci.User_id
        right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
        WHERE ci.Microcat_id IN
        (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
        and subcategory_slug = 'vakansii' )
        AND cpe.event_time >= '{end_date}'::date-180
        and cpe.event_time <= '{end_date}'::date
        and cpe.is_revenue = true 
        and cu.user_id not in (select user_id from dma.current_user where isTest)
        and cu.user_id in ( 
            {sql_active_sellers}
        )
        and ci.isDead = false
        and ci.Param1 = '{item[0]}'
        group by 1
        """
    
    f_name = transliterate(item[0])
    print(f_name, collect_feature(sql, f_name))


# In[33]:


# все типы вакансий селлеров
with vertica_python.connect(**conn_info) as conn:

    sql = f"""
    select  distinct ci.Param2
    from DMA.current_user cu
    right join DMA.current_item ci ON cu.User_id = ci.User_id
    right join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
    WHERE ci.Microcat_id IN
    (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
    and subcategory_slug = 'vakansii')
    AND cpe.event_time >= '{end_date}'::date-180
    and cpe.event_time <= '{end_date}'::date
    and cpe.is_revenue = true 
    and cu.user_id not in (select user_id from dma.current_user where isTest)
    and cu.user_id in ( 
            {sql_active_sellers}
        )
    and ci.isDead = false
    """

    cur = conn.cursor()
    cur.execute(sql)
    Param2 = cur.fetchall()


# In[34]:


Param2


# In[35]:


# количество вакансий данного типа по селлерам

for item in Param2:
    
    if not item[0]:
        continue
    
    sql = f"""
        select  cu.User_id, count(ci.Param2)::float as cnt
        from DMA.current_user cu
        left join DMA.current_item ci ON cu.User_id = ci.User_id
        left join dma.current_payment_events cpe ON cpe.Item_id = ci.Item_id
        WHERE ci.Microcat_id IN
        (SELECT distinct Microcat_id FROM DMA.current_microcategories where vertical = 'Jobs'
        and subcategory_slug = 'vakansii' )
        AND cpe.event_time >= '{end_date}'::date-180
        and cpe.event_time <= '{end_date}'::date
        and cpe.is_revenue = true 
        and cu.user_id not in (select user_id from dma.current_user where isTest)
        and cu.user_id in ( 
            {sql_active_sellers}
        )
        and ci.isDead = false
        and ci.Param2 = '{item[0]}'
        group by 1
        """
    
    f_name = transliterate(item[0])
    print(f_name, collect_feature(sql, f_name))


# In[36]:


# collect clickstream

import cs_events

cs_extids = {e[3]:transliterate(e[0]) for e in cs_events.cs_events}  # {eid: event_name_en}


# In[37]:


# collect cs raw
with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()

    cur.execute('drop table if exists ivolga.preasd_t2')

    cur.execute(f'create table ivolga.preasd_t2 (user_id int, eid int, cnt int)')

    sql = f"""
    insert into ivolga.preasd_t2 (user_id, eid, cnt)
    select user_id, eid, count(*)::int
    from DMA.jobs_employers_clickhouse_history
    where user_id in 
    (
        {sql_active_sellers}
    )
    and event_date >= '{end_date_}'::date - 90
    and event_date <= '{end_date_}'::date
    and eid in {tuple(cs_extids.keys())}
    group by user_id, eid
    """
    
    cur.execute(sql)

    cur.execute('commit;')
    
    


# In[38]:


conditions = ' case '
for eid, name in cs_extids.items():
    conditions += f""" when eid = '{eid}' then '{name}_csf' 
    """
conditions += " else 'None' end"


# In[39]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()

    cur.execute(f'alter table ivolga.preasd_t2 add column IF NOT EXISTS e_name varchar')
    
    sql = f"""
    update ivolga.preasd_t2 set e_name = {conditions};
    """
    
    cur.execute(sql)

    cur.execute('commit;')


# In[40]:


# pivot table cs

cs_data = vdf_from_relation(f"(select * from ivolga.preasd_t2) x")


# In[41]:


#cs_data


# In[42]:


cs_data1 = cs_data.pivot(index='user_id', 
                 columns='e_name', 
                 values='cnt', 
                 aggr='sum')


# In[43]:


#cs_data1


# In[44]:


data = vdf_from_relation(f"(select * from ivolga.preasd_t3) x")


# In[45]:


#data


# In[46]:


data1 = data.join(cs_data1, 
                 how = "left",
                 on = {"user_id": "user_id"},
                 expr1 = ["*"],
                 expr2 = [i.replace('"', '') for i in list(cs_data1.get_columns())][1:]
                 )


# In[ ]:


data1.fillna(val = {e: 0 for e in cs_data1.get_columns()})


# In[ ]:


#for c in data1.get_columns():
#    data1[c].fillna(0)


# In[ ]:


data1


# In[ ]:


#data1


# In[ ]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    cur.execute('drop table if exists ivolga.preasd_dataset')
    
    cur.execute('commit;')
    
data1.to_db('"ivolga"."preasd_dataset"', relation_type = "table")

    


# In[ ]:





# In[ ]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    
    cur.execute("""
    create table if not exists IVOLGA.runlog (
        date_time TIMESTAMP DEFAULT sysdate,
        task VARCHAR,
        operation VARCHAR,
        comment VARCHAR(1000)
    )
    """)
    cur.execute('commit;')


# In[ ]:


duration = datetime.datetime.now() - start_time


# In[ ]:


run_log_comment = json.dumps({
    'dataset_length': len(data1),
    'asd': len(data1[data1['is_asd'] == 1]),
    'non_asd': len(data1[data1['is_asd'] == 0]),
    'duration': str(duration),
})


# In[ ]:


print(run_log_comment)


# In[ ]:


# run log

"""
create table if not exists IVOLGA.runlog (
    date_time TIMESTAMP DEFAULT sysdate,
    task VARCHAR,
    operation VARCHAR,
    comment VARCHAR(1000)
)
"""

with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    
    sql = f"""
    insert into IVOLGA.runlog (task, operation, comment) values ('preasd', 'collect_dataset', 
    '{run_log_comment}')
    """

    cur.execute(sql)
    cur.execute('commit;')


# In[ ]:


with vertica_python.connect(**conn_info) as conn:
    cur = conn.cursor()
    
    sql = """
        drop table if exists ivolga.preasd_t1
        """
    cur.execute(sql)

    sql = """
        drop table if exists ivolga.preasd_t2
        """
    cur.execute(sql)

    sql = """
        drop table if exists ivolga.preasd_t3
        """
    cur.execute(sql)

    cur.execute('commit;')


# In[60]:


print(duration)


# In[ ]:


1+1

