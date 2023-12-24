import random
import re
from datetime import datetime

import numpy as np
import pandas as pd


class Transformer:
    def transform(self, data: pd.DataFrame):
        '''
        Implementasi berupa prosedur (Facade pattern) untuk men-transform test dataset `*_test.csv` ke dalam
        bentuk yang dapat diproses secara langsung oleh model machine learning yang telah dibuat.
        '''
        pass


class DatasetUtils:
    '''
    Utility class untuk dataset hasil merger dari `matches.csv`, `games.csv`, dan `scores.csv` 
    '''

    def __init__(self):
        self.classification_features = {}
        self.regression_features = {}

    def remove_redundant_attr(self):
        pass

    def remove_meta_attr(self):
        pass

    def merge_datasets(self, df_matches, df_games, df_scores):
        df_matches_games = pd.merge(df_matches, df_games, on=[
                                    'MatchID', 'Team1ID', 'Team2ID'])
        df_matches_games_scores = pd.merge(
            df_matches_games, df_scores, on=['GameID'])
        return df_matches_games_scores

    def classification_select(self, key):
        '''
        Method untuk memilih fitur yang akan digunakan dalam permasalahan klasifikasi
        '''
        pass

    def regression_select(self, key):
        '''
        Method untuk memilih fitur yang akan digunakan dalam permasalahan regressi
        '''
        pass


class MatchesUtils(Transformer):
    '''
    Utility class untuk dataset `matches.csv`
    '''

    def transform(self, data: pd.DataFrame):
        pass

    def patch_transform(self, df: pd.DataFrame):
        def patch_transform(x):
            regex = r"(?<=Patch\s)\d"
            match = re.search(regex, x)
            return match.group() if match else x

        return df['Patch'].map(patch_transform, na_action='ignore')

    def patch_impute(self, df_matches: pd.DataFrame, df_patch_date_range: pd.DataFrame):
        '''
        Imputasi Kolom `Patch` berdasarkan tanggal `Date` dari suatu pertandingan
        '''

        def patch_imputer(row):
            if not pd.isna(row['Patch']):
                return row['Patch']

            row_date_obj = datetime.strptime(row['Date'], "%Y-%m-%d")

            # Pengecekan `Date` berdasarkan nilai paling min
            lowest_patch_min_date = df_patch_date_range['Date']['min'].min()
            lowest_patch_min_date_obj = datetime.strptime(
                lowest_patch_min_date, "%Y-%m-%d")

            if row_date_obj < lowest_patch_min_date_obj:
                return df_patch_date_range['Date']['min'].idxmin()

            # Pengecekan `Date` berdasarkan range `df_patch_date_range`
            for patch, date_range in df_patch_date_range.iterrows():
                patch_min_date = date_range['Date']['min']
                patch_max_date = date_range['Date']['max']

                patch_min_date_obj = datetime.strptime(
                    patch_min_date, "%Y-%m-%d")
                patch_max_date_obj = datetime.strptime(
                    patch_max_date, "%Y-%m-%d")

                # Pengecekan `Date` dari suatu pertandingan untun menentukan Patch
                if patch_min_date_obj <= row_date_obj <= patch_max_date_obj:
                    return patch

            return row['Patch']

        return df_matches[['Date', 'Patch']].apply(
            patch_imputer, axis='columns')

    def date_transform(self, df: pd.DataFrame):
        def date_transform(x):
            date, *_ = x.split()
            return date

        return df['Date'].map(date_transform, na_action='ignore')

    def date_transform_remove_day(self, df: pd.DataFrame):
        def date_transform(x):
            year, month, _ = x.split('-')
            return f'{year}-{month}'

        return df['Date'].map(date_transform, na_action='ignore')


class GamesUtils(Transformer):
    '''
    Utility class untuk dataset `games.csv`
    '''

    def transform(self, data: pd.DataFrame):
        pass

    def econ_impute(self, df_games: pd.DataFrame, df_team_econ_median: pd.DataFrame, srs_econ_median: pd.Series):
        def econ_impute(row):
            if not row.isna().any():
                return row

            if row[['Team1ID', 'Team1_Eco', 'Team1_SemiEco', 'Team1_SemiBuy', 'Team1_FullBuy']].isna().any():
                team1_econ_info = df_team_econ_median.loc[int(row['Team1ID'])]

                if team1_econ_info.isna().any():
                    team1_econ_info = srs_econ_median

                row['Team1_Eco'] = team1_econ_info['Team_Eco']
                row['Team1_SemiEco'] = team1_econ_info['Team_SemiEco']
                row['Team1_SemiBuy'] = team1_econ_info['Team_SemiBuy']
                row['Team1_FullBuy'] = team1_econ_info['Team_FullBuy']
            if row[['Team2ID', 'Team2_Eco', 'Team2_SemiEco', 'Team2_SemiBuy', 'Team2_FullBuy']].isna().any():
                team2_econ_info = df_team_econ_median.loc[int(row['Team2ID'])]

                if team2_econ_info.isna().any():
                    team2_econ_info = srs_econ_median

                row['Team2_Eco'] = team2_econ_info['Team_Eco']
                row['Team2_SemiEco'] = team2_econ_info['Team_SemiEco']
                row['Team2_SemiBuy'] = team2_econ_info['Team_SemiBuy']
                row['Team2_FullBuy'] = team2_econ_info['Team_FullBuy']

            return row

        return df_games[['Team1ID', 'Team1_Eco', 'Team1_SemiEco', 'Team1_SemiBuy', 'Team1_FullBuy', 'Team2ID', 'Team2_Eco', 'Team2_SemiEco', 'Team2_SemiBuy', 'Team2_FullBuy']].apply(econ_impute, axis='columns')


class ScoresUtils(Transformer):
    '''
    Utility class untuk dataset `scores.csv`
    '''

    def transform(self, data: pd.DataFrame):
        pass

    def cek_null(self, df: pd.DataFrame):
        col_na = df.isnull().sum().sort_values(ascending=False)
        percent = col_na / len(df) * 100

        missing_data = pd.concat(
            [col_na, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data[missing_data['Total'] > 0])

    def player_id_impute(self, df: pd.DataFrame):
        id_counter = df['PlayerID'].max() + 1
        player_ids = {}

        def player_id_imputer(row):
            nonlocal id_counter, player_ids

            if pd.isnull(row['PlayerID']):
                if row['PlayerName'] in player_ids:
                    return player_ids[row['PlayerName']]

                player_ids[row['PlayerName']] = id_counter
                id_counter += 1
                return player_ids[row['PlayerName']]

            return row['PlayerID']

        return df[['PlayerID', 'PlayerName']].apply(
            player_id_imputer, axis='columns')

    def agent_impute(self, df: pd.DataFrame, df_merge, srs_most_used_hero_by_player):
        random.seed(1000)
        df_game_id_patch = df_merge[['GameID', 'Patch']]

        def agent_imputer(row):
            patch_nums = df_game_id_patch.loc[df_game_id_patch['GameID']
                                              == row['GameID'], 'Patch']

            if (not pd.isnull(row['Agent'])) or len(patch_nums) == 0:
                return row['Agent']

            # Imputasi berdasarkan agent yang paling sering dimainkan
            agent = srs_most_used_hero_by_player[row['PlayerID'],
                                                 patch_nums.iloc[0]]

            if isinstance(agent, str):
                return agent
            elif isinstance(agent, np.ndarray) and len(agent) >= 2:
                return agent[random.randint(0, len(agent)-1)]

            return row['Agent']

        return df[['PlayerID', 'Agent', 'GameID']].apply(
            agent_imputer, axis='columns')

    def acs_impute(self, df_scores: pd.DataFrame, acs_median: pd.DataFrame):
        '''
        Imputasi Kolom 'ACS' berdasarkan Agent yang dipilih
        '''
        def acs_imputer(row):
            if not pd.isna(row['ACS']):
                return row['ACS']

            row_agent_obj = row['Agent']

            # Pengecekan Agent untuk menentukan nilai ACS
            agent_row = acs_median.loc[acs_median['Agent'] == row_agent_obj]

            if not agent_row.empty:
                return agent_row['ACS'].iloc[0]

            return row['ACS']

        return df_scores[['Agent', 'ACS']].apply(
            acs_imputer, axis='columns')

    def team_abbreviation_impute(self, df_scores: pd.DataFrame, df_game_team_agent: pd.DataFrame):
        def team_abbreviation_impute(row):
            if not row.isna().any():
                return row

            team_agent_info = df_game_team_agent.loc[int(row['GameID'])]
            team_abbr = team_agent_info['TeamAbbreviation']
            team_agent = team_agent_info['Agent']

            try:
                team1, team2 = set(team_abbr)
                team1_count = team_abbr.count(team1)
                team2_count = team_abbr.count(team2)

                if team1_count == 5 and team2_count == 4:
                    row['TeamAbbreviation'] = team2
                elif team1_count == 4 and team2_count == 5:
                    row['TeamAbbreviation'] = team1
                elif row['Agent'] in team_agent:
                    enemy_agent_index = team_agent.index(row['Agent'])
                    enemy_team = team_agent[enemy_agent_index]
                    row['TeamAbbreviation'] = team1 if enemy_team == team2 else team2
                else:
                    row['TeamAbbreviation'] = team1 if team1_count < team2_count else team2
            except:
                team = set(team_abbr).pop()
                team_count = team_abbr.count(team)

                if team_count < 5:
                    row['TeamAbbreviation'] = team
            finally:
                return row

        return df_scores[['GameID', 'TeamAbbreviation', 'Agent']].apply(team_abbreviation_impute, axis='columns')

    def merge_player_id_and_name(self, df_scores):
        def merge_player_id_and_name(row):
            return f'{row['PlayerName']}#{int(row['PlayerID'])}'
        return df_scores[['PlayerID', 'PlayerName']].apply(merge_player_id_and_name, axis='columns')


# Objek utility
util_matches = MatchesUtils()
util_games = GamesUtils()
util_scores = ScoresUtils()
util_dataset = DatasetUtils()
