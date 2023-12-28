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


class DatasetUtils(Transformer):
    '''
    Utility class untuk dataset hasil merger dari `matches.csv`, `games.csv`, dan `scores.csv` 
    '''

    def transform(self, df_matches, df_games, df_scores, df_patch_agent=None):
        df = self.merge_datasets(df_matches, df_games, df_scores)

        if df_patch_agent is not None:
            df['Total_Agent'] = self.create_total_agent(df, df_patch_agent)

        return df

    def merge_datasets(self, df_matches, df_games, df_scores):
        df_matches_games = pd.merge(df_matches, df_games, on=[
                                    'MatchID', 'Team1ID', 'Team2ID'])
        df_matches_games_scores = pd.merge(
            df_matches_games, df_scores, on=['GameID'])
        return df_matches_games_scores

    def remove_meta_attr_classification(self, df: pd.DataFrame):
        meta_attr = [
            'No_x', 'MatchID', 'EventID', 'EventName',
            'Team1ID', 'Team2ID', 'Team1_x', 'Team2_x',
            'Team1_MapScore', 'Team2_MapScore', 'No_y',
            'GameID', 'Team1_y', 'Team2_y', 'Winner', 'Team1_Eco',
            'Team1_SemiEco', 'Team1_SemiBuy', 'Team1_FullBuy',
            'Team1_TotalRounds', 'Team2_Eco', 'Team2_SemiEco',
            'Team2_SemiBuy', 'Team2_FullBuy', 'Team2_TotalRounds',
            'No', 'KAST_Percent', 'PlayerName', 'PlayerID', 'TeamAbbreviation', 'EventStage',
            'Num_2Ks', 'Num_3Ks', 'Num_4Ks', 'Num_5Ks', 'OnevOne', 'OnevTwo', 'OnevThree', 'OnevFour', 'OnevFive',
        ]

        if 'ACS' in df.columns:
            meta_attr.append('ACS')

        return df.drop(meta_attr, axis='columns')

    def remove_meta_attr_regression(self, df: pd.DataFrame):
        meta_attr = [
            'No_x', 'MatchID', 'EventID', 'EventName', 'Date', 'Patch', 'Map', 'Team_MapScore',
            'Team1ID', 'Team2ID', 'Team1_x', 'Team2_x',
            'Team1_MapScore', 'Team2_MapScore', 'No_y',
            'GameID', 'Team1_y', 'Team2_y', 'Winner', 'Team1_Eco',
            'Team1_SemiEco', 'Team1_SemiBuy', 'Team1_FullBuy',
            'Team1_TotalRounds', 'Team2_Eco', 'Team2_SemiEco',
            'Team2_SemiBuy', 'Team2_FullBuy', 'Team2_TotalRounds',
            'No', 'KAST_Percent', 'PlayerName', 'PlayerID', 'TeamAbbreviation', 'EventStage',
            'Num_2Ks', 'Num_3Ks', 'Num_4Ks', 'Num_5Ks', 'OnevOne', 'OnevTwo', 'OnevThree', 'OnevFour', 'OnevFive',
            'Kills', 'Deaths', 'Assists', 'PlusMinus', 'ADR'
        ]

        if 'Agent' in df.columns:
            meta_attr.append('Agent')

        return df.drop(meta_attr, axis='columns')

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

    def create_total_agent(self, df: pd.DataFrame, df_patch_agent: pd.DataFrame):
        def create_total_agent(x):
            return df_patch_agent.loc[x][0]

        return df['Patch'].map(create_total_agent)


class MatchesUtils(Transformer):
    '''
    Utility class untuk dataset `matches.csv`
    '''

    def transform(self, data: pd.DataFrame, df_patch_date_range: pd.DataFrame):
        df = data.copy()
        df['Date'] = self.date_transform(df)
        df['Patch'] = self.patch_impute(df, df_patch_date_range)
        df['Patch'] = self.patch_transform(df)
        df['Date'] = self.date_transform_remove_day(df)
        df['Date'] = self.date_diff_transform(df)
        df['Team_MapScore'] = self.create_team_mapscore_avg(df)
        return df

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

    def date_diff_transform(self, df: pd.DataFrame):
        date_valorant_published = '2020-06-02'
        date_subtrahend = datetime.strptime(
            date_valorant_published, '%Y-%m-%d')

        def date_diff_transform(x):
            date_minuend = datetime.strptime(x, '%Y-%m')
            time_difference = (date_minuend - date_subtrahend).days
            return time_difference

        return df['Date'].apply(date_diff_transform)

    def create_team_mapscore_avg(self, df_matches: pd.DataFrame):
        def create_team_mapscore(row):
            return (row['Team1_MapScore'] + row['Team2_MapScore'])/2

        return df_matches[['Team1_MapScore', 'Team2_MapScore']].apply(create_team_mapscore, axis='columns')


class GamesUtils(Transformer):
    '''
    Utility class untuk dataset `games.csv`
    '''

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        df['Team_Eco'] = self.create_team_eco_avg(df)
        df['Team_SemiEco'] = self.create_team_semieco_avg(df)
        df['Team_SemiBuy'] = self.create_team_semibuy_avg(df)
        df['Team_FullBuy'] = self.create_team_fullbuy_avg(df)
        df['Team_TotalRounds'] = self.create_team_total_round_avg(df)
        return df

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

    def create_team_eco_avg(self, df_games: pd.DataFrame):
        def create_team_eco_avg(row):
            return (row['Team1_Eco'] + row['Team2_Eco'])/2

        return df_games[['Team1_Eco', 'Team2_Eco']].apply(create_team_eco_avg, axis='columns')

    def create_team_semieco_avg(self, df_games: pd.DataFrame):
        def create_team_semieco_avg(row):
            return (row['Team1_SemiEco'] + row['Team2_SemiEco'])/2

        return df_games[['Team1_SemiEco', 'Team2_SemiEco']].apply(create_team_semieco_avg, axis='columns')

    def create_team_semibuy_avg(self, df_games: pd.DataFrame):
        def create_team_semibuy_avg(row):
            return (row['Team1_SemiBuy'] + row['Team2_SemiBuy'])/2

        return df_games[['Team1_SemiBuy', 'Team2_SemiBuy']].apply(create_team_semibuy_avg, axis='columns')

    def create_team_fullbuy_avg(self, df_games: pd.DataFrame):
        def create_team_fullbuy_avg(row):
            return (row['Team1_FullBuy'] + row['Team2_FullBuy'])/2

        return df_games[['Team1_FullBuy', 'Team2_FullBuy']].apply(create_team_fullbuy_avg, axis='columns')

    def create_team_total_round_avg(self, df_games: pd.DataFrame):
        def create_team_fullbuy_avg(row):
            return (row['Team1_TotalRounds'] + row['Team2_TotalRounds'])/2

        return df_games[['Team1_TotalRounds', 'Team2_TotalRounds']].apply(create_team_fullbuy_avg, axis='columns')


class ScoresUtils(Transformer):
    '''
    Utility class untuk dataset `scores.csv`
    '''

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        df['Num_Ks'] = self.create_numks(df)
        df['OnevX'] = self.create_onevx(df)
        return df

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

    def create_numks(self, df_scores: pd.DataFrame):
        def create_numks(row):
            return (2*row['Num_2Ks'] + 3*row['Num_3Ks'] + 4*row['Num_4Ks'] + 5*row['Num_5Ks'])/14
        return df_scores[['Num_2Ks', 'Num_3Ks', 'Num_4Ks', 'Num_5Ks']].apply(create_numks, axis='columns')

    def create_onevx(self, df_scores: pd.DataFrame):
        def create_onevx(row):
            return (1*row['OnevOne']+2*row['OnevTwo'] + 3*row['OnevThree'] + 4*row['OnevFour'] + 5*row['OnevFive'])/15
        return df_scores[['OnevOne', 'OnevTwo', 'OnevThree', 'OnevFour', 'OnevFive']].apply(create_onevx, axis='columns')


# Objek utility
util_matches = MatchesUtils()
util_games = GamesUtils()
util_scores = ScoresUtils()
util_dataset = DatasetUtils()
