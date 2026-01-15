# import botbuster
import utils_model_helper

import os, json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from icecream import ic


def read_models():
    try:
        models_folder = "models"
        global desc_model, username_model, screenname_model, posts_model, posts_metadata_model, user_metadata_model
        
        desc_model = joblib.load(os.path.join(models_folder, 'description.pkl'))
        username_model = joblib.load(os.path.join(models_folder, 'user_name.pkl'))
        screenname_model = joblib.load(os.path.join(models_folder, 'screen_name.pkl'))
        posts_model = joblib.load(os.path.join(models_folder, 'posts.pkl'))
        posts_metadata_model = joblib.load(os.path.join(models_folder, 'posts_metadata.pkl'))
        user_metadata_model = joblib.load(os.path.join(models_folder, 'user_metadata.pkl'))
        return 1
    
    except:
        return 0

def check_known_expert(username, screenname, description, is_verified):
    if is_verified == True:
        return 'human'
    
    if username != None and username != '':
        username = username.lower()
        if 'bot' in username:
            return 'bot'
        
    if description is not None and description != '':
        description = description.lower()
        if 'bot' in description:
            return 'bot'
        
    if screenname is not None and screenname != '':
        screenname = screenname.lower()
        if 'bot' in screenname:
            return 'bot'
    
    return None

# Helper functions to get probabilities of each type
def get_description_prob(description):
    if description == None or description == '':
        return None
    
    description = description.lower()
    
    desc_arr = [{'description': description}]
    df = pd.DataFrame(desc_arr)
    df['description_cleaned'] = df['description'].apply(utils_model_helper.preprocess_text)
    df_test = df[['description_cleaned']]
    predictions = desc_model.predict_proba(df_test)
    
    return predictions

def get_username_prob(username):
    if username == None or username == '':
        return None
    
    username = username.lower()
    
    username_arr = [{'user_name': username}]
    df = pd.DataFrame(username_arr)
    df = utils_model_helper.transform_df(df, 'user_name')
    df_test = df[utils_model_helper.username_cols]
    predictions = username_model.predict_proba(df_test)
    return predictions

def get_screenname_prob(screenname):   
    if screenname == None or screenname == '':
        return None
    
    screenname = screenname.lower()
    
    screenname_arr = [{'screen_name': screenname}]
    df = pd.DataFrame(screenname_arr)
    df = utils_model_helper.transform_df(df, 'screen_name')
    df_test = df[utils_model_helper.screenname_cols]
    predictions = screenname_model.predict_proba(df_test)
    return predictions

def get_posts_df_reddit(post):
    title = post.get("title") or ""
    selftext = post.get("selftext") or ""
    text = (title + "\n" + selftext).strip()
    if not text:
        return None

    text_cleaned = utils_model_helper.preprocess_text(text)
    if not text_cleaned:
        return None

    # IMPORTANT: posts_model expects ONLY ONE feature column
    return pd.DataFrame([{"text_cleaned": text_cleaned}])


def get_usermetadata_prob(df):    
    df = df.fillna(-1)
    df_temp = df[(df['followers_count'] == -1) & (df['listed_count'] == -1) & \
                        (df['protected'] == -1) & (df['verified'] == -1) & \
                        (df['following_count'] == -1) & (df['like_count'] == -1) ]
    
    cond = df['userid'].isin(df_temp['userid'])
    df_temp_final = df.drop(df[cond].index)
    
    if len(df_temp_final) == 0:
        print('No metadata')
        return None
    
    df_test = df_temp_final[utils_model_helper.usermetadata_cols]
    predictions = user_metadata_model.predict_proba(df_test)
    return predictions

def get_posts_prob(df_posts):
    if df_posts is None or len(df_posts) == 0:
        return None

    df = df_posts.fillna(-1)

    # Figure out what columns the trained model expects
    expected_cols = None

    # Common for sklearn Pipelines with a ColumnTransformer inside
    try:
        prep = posts_model.named_steps.get("preprocessor")
        if prep is not None and hasattr(prep, "feature_names_in_"):
            expected_cols = list(prep.feature_names_in_)
    except Exception:
        pass

    # Common for models trained directly on a DataFrame
    if expected_cols is None:
        try:
            if hasattr(posts_model, "feature_names_in_"):
                expected_cols = list(posts_model.feature_names_in_)
        except Exception:
            pass

    # Fallback: just use whatever we have (but this might still fail)
    if expected_cols is None:
        expected_cols = list(df.columns)

    # Ensure all expected columns exist
    for c in expected_cols:
        if c not in df.columns:
            df[c] = -1

    df_test = df[expected_cols]

    return posts_model.predict_proba(df_test)


def compute_botbuster_prob(out_fh, userid, username, screenname, description, is_verified, df_posts, df_usermetadata):
    known_data_expert = check_known_expert(username, screenname, description, is_verified) 
    if known_data_expert is not None:
        if known_data_expert == 'bot':
            out_fh.write(f'{userid},0,1,True\n')
            
            return
        elif known_data_expert == 'human':
            out_fh.write(f'{userid},1,0,False\n')
            
            return
                            
    prob_arr = [0, 0]
    count = 0
            
    username_prob = get_username_prob(username)
    if username_prob is not None:
        prob_arr += username_prob[0]
        count += 1   
                
    screenname_prob = get_screenname_prob(screenname)
    if screenname_prob is not None:
        prob_arr += screenname_prob[0]
        count += 1

    description_prob = get_description_prob(description)
    if description_prob is not None:
        prob_arr += description_prob[0]
        count += 1

    if df_posts is not None:
        posts_prob = get_posts_prob(df_posts)
        if posts_prob is not None:
            prob_arr += posts_prob[0]
            count += 1

    if df_usermetadata is not None:
        usermetadata_prob = get_usermetadata_prob(df_usermetadata)
        if usermetadata_prob is not None:
            prob_arr += usermetadata_prob[0]
            count += 1
                
    ic(prob_arr)

    if count == 0:
        out_fh.write(f'{userid},0.5,0.5,False\n')
        return

    prob_arr_div = prob_arr / count

    bot_prob = prob_arr_div[0]
    human_prob = prob_arr_div[1]

    overall_bot_prob = max(prob_arr_div)
    max_index = np.where(prob_arr_div == overall_bot_prob)[0][0]

    botornot = False
    if max_index == 0:
        botornot = True
    elif max_index == 1:
        botornot = False
    
    out_fh.write(f'{userid},{human_prob},{bot_prob},{botornot}\n')

    return

def get_posts_df_v1(line_json):
    if line_json['full_text'] == '':
        return None
    
    text_cleaned = utils_model_helper.preprocess_text(line_json['full_text'])
    if text_cleaned == '':
        return None
    
    user_post_arr = [{'text_cleaned': text_cleaned,
                      'post_like_count': line_json['favorite_count'],
                      'post_retweet_count': line_json['retweet_count'],
                      'post_reply_count': -1,
                      'post_quote_count': -1
    }]
    
    df = pd.DataFrame(user_post_arr)
    return df

def get_posts_df_v2(tweet):
    if tweet['text'] == '' or tweet['text'] == None:
        return None
    
    text_cleaned = utils_model_helper.preprocess_text(tweet['text'])
    if text_cleaned == '':
        return None
    
    user_post_arr = [{'text_cleaned': text_cleaned,
                      'post_like_count': tweet['public_metrics']['like_count'],
                      'post_retweet_count': tweet['public_metrics']['retweet_count'],
                      'post_reply_count': tweet['public_metrics']['reply_count'],
                      'post_quote_count': tweet['public_metrics']['quote_count']
    }]
    
    df = pd.DataFrame(user_post_arr)
    return df


def run_botbuster_reddit(in_filename):

    out_filename = in_filename.replace('.json','_bots.json')
    ic(in_filename)
    ic(out_filename)

    # set up file header for writing
    out_fh =  open(out_filename, 'w', encoding='utf-8')
    out_fh.write('userid,humanprobability,botprobability,botornot\n')

    with open(in_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line)

            # pull author identifiers safely first
            username = line_json.get("author_fullname") or line_json.get("author_id")
            screenname = line_json.get("author")

            # if we truly have no author info, skip
            if not username and not screenname:
                continue

            # now set userid based on AUTHOR (not post id)
            userid = username or screenname

            is_verified = None
            description = None

            df_posts = get_posts_df_reddit(line_json)
            df_usermetadata = None

            compute_botbuster_prob(
                out_fh,
                userid,
                username,
                screenname,
                description,
                is_verified,
                df_posts,
                df_usermetadata
            )


    out_fh.close()


if __name__ == "__main__":
    # Default to the repo-level test_data folder
    BASE_DIR = Path(__file__).resolve().parent
    file_to_analyze = str((BASE_DIR.parent / "test_data" / "test_reddit.json").resolve())
    out_fh = None

    model_reading = read_models()
    if model_reading != 1:
        raise SystemExit("Failed to load models; check that the 'models' folder is present and dependencies match.")
    botbuster_run = run_botbuster_reddit(file_to_analyze)
