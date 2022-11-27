import base64
import json
from io import BytesIO

import cmasher as cmr
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import unicodedata
import matplotlib.ticker as plticker
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch, Sbopen
from mplsoccer.pitch import Pitch
from scipy.ndimage import gaussian_filter


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_teams(game_code):
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')
    team_1 = df['team_name'].unique()[0]
    team_2 = df['team_name'].unique()[1]
    return [team_1, team_2]


def get_players(game_code, team):
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')

    mask_2 = df.loc[df['team_name'] == team]
    player_names_2 = mask_2['player_name'].dropna().unique()
    return player_names_2


# Define five functions for five activities from drop-down menu
# Pass plot function
def pressure_map(game_code, player, team):
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.draw()
    plt.title('Давление: ' + player)
    pitch_height = 80
    pitch_width = 120
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')

    # Replace non-unicode characters in players names
    df['player_name'] = df['player_name'].astype(str)
    df['player_name'] = df['player_name'].apply \
        (lambda val: unicodedata.normalize('NFC', val).encode('ascii', 'ignore').decode('utf-8'))
    df['player_name'] = df['player_name'].replace('nan', np.nan)

    df_pressure = df.loc[(df['player_name'] == player) & (df['type_name'] == 'Pressure')]
    location = df_pressure['location'].tolist()
    dot_size = 2
    color = 'orange' if team == 'Есперион' else 'red'
    if team == 'Есперион':
        x = np.array([el[0] for el in location])
        y = pitch_height - np.array([el[1] for el in location])
    else:
        x = pitch_width - np.array([el[0] for el in location])
        y = np.array([el[1] for el in location])
    for x, y in zip(x, y):
        dot = plt.Circle((x, y), dot_size, color=color, alpha=0.5)
        ax.add_patch(dot)
    # Display Pitch
    graph = get_graph()
    return graph


def pass_map(game_code, player, team):
    pitch_height = 80
    pitch_width = 120
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.draw()
    plt.title('Пасы: ' + player)
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')

    df_pass = df.loc[(df['player_name'] == player) & (df['type_name'] == 'Pass')]
    location = df_pass['location'].tolist()
    pass_end_location = df_pass['pass_end_location'].tolist()
    color = 'orange' if team == 'Есперион' else 'red'
    if team == 'Есперион':
        x1 = np.array([el[0] for el in location])
        y1 = pitch_height - np.array([el[1] for el in location])
        x2 = np.array([el[0] for el in pass_end_location])
        y2 = pitch_height - np.array([el[1] for el in pass_end_location])
    else:
        x1 = pitch_width - np.array([el[0] for el in location])
        y1 = np.array([el[1] for el in location])
        x2 = pitch_width - np.array([el[0] for el in pass_end_location])
        y2 = np.array([el[1] for el in pass_end_location])
    u = x2 - x1
    v = y2 - y1
    ax.quiver(x1, y1, u, v, color=color, width=0.003, headlength=4.5)

    graph = get_graph()
    return graph


def pass1(game_code, player, team, side):
    pitch_length_X = 120
    pitch_width_Y = 80

    teamA = team  # <--- adjusted here

    ## calling the function to create a pitch map
    ## yards is the unit for measurement and
    ## gray will be the line color of the pitch map
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.draw()  # < moved into for loop

    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_').assign(match_id=game_code[:-5])
    teamB = [x for x in list(df['team_name'].unique()) if x != teamA][0]  # <--- get other team name

    ## making the list of all column names
    column = list(df.columns)

    ## all the type names we have in our dataframe
    all_type_name = list(df['type_name'].unique())

    ## creating a data frame for pass
    ## and then removing the null values
    ## only listing the player_name in the dataframe
    pass_df = df.loc[df['type_name'] == 'Pass', :].copy()
    pass_df.dropna(inplace=True, axis=1)
    pass_df = pass_df.loc[pass_df['player_name'] == player, :]

    ## creating a data frame for ball receipt
    ## removing all the null values
    ## and only listing Barcelona players in the dataframe
    breceipt_df = df.loc[df['type_name'] == 'Ball Receipt*', :].copy()
    breceipt_df.dropna(inplace=True, axis=1)
    breceipt_df = breceipt_df.loc[breceipt_df['team_name'] == team, :]

    pass_comp, pass_no = 0, 0
    ## pass_comp: completed pass
    ## pass_no: unsuccessful pass

    ## iterating through the pass dataframe
    for row_num, passed in pass_df.iterrows():

        if passed['player_name'] == player:
            ## for away side
            x_loc = passed['location'][0]
            y_loc = passed['location'][1]

            pass_id = passed['id']

            events_list = [item for sublist in breceipt_df['related_events'] for item in sublist]
            if pass_id in events_list:
                ## if pass made was successful
                color = '#ffd23f'
                pass_comp += 1
            else:
                ## if pass made was unsuccessful
                color = '#ff3f46'
                pass_no += 1

                ## plotting circle at the player's position

            ## parameters for making the arrow
            pass_x = 120 - passed['pass_end_location'][0]
            pass_y = passed['pass_end_location'][1]
            dx = ((pitch_length_X - x_loc) - pass_x)
            dy = y_loc - pass_y

            if side == 'home':
                shot_circle = plt.Circle((x_loc, pitch_width_Y - y_loc), radius=2, color=color)
                shot_circle.set_alpha(alpha=0.2)
                ax.add_patch(shot_circle)
                pass_arrow = plt.arrow(x_loc, pitch_width_Y - y_loc, dx, dy, width=0.1, head_length=2.5, color=color,
                                       head_width=1)
                ax.add_patch(pass_arrow)
            elif side == 'away':
                shot_circle = plt.Circle((pitch_length_X - x_loc, y_loc), radius=2, color=color)
                shot_circle.set_alpha(alpha=0.2)
                ax.add_patch(shot_circle)
                pass_arrow = plt.arrow(pitch_length_X - x_loc, y_loc, -dx, -dy, width=0.1, head_length=2.5, color=color,
                                       head_width=1)
                ax.add_patch(pass_arrow)

            ## making an arrow to display the pass

            ## adding arrow to the plot

        ## computing pass accuracy
    pass_acc = (pass_comp / (pass_comp + pass_no)) * 100
    pass_acc = str(round(pass_acc, 2))

    plt.title('Пасы: {}. Точность передач: {}'.format(player, pass_acc))  # <-- change to title

    graph = get_graph()
    return graph


def ball_receipt_map(game_code, player, team):
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.draw()
    plt.title('Приём мяча: ' + player)
    pitch_height = 80
    pitch_width = 120
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')

    df_ball_rec = df.loc[(df['player_name'] == player) & (df['type_name'] == 'Ball Receipt*')]
    location = df_ball_rec['location'].tolist()
    dot_size = 1
    color = 'orange' if team == 'Есперион' else 'red'
    if team == 'Есперион':
        x = np.array([el[0] for el in location])
        y = pitch_height - np.array([el[1] for el in location])
    else:
        x = pitch_width - np.array([el[0] for el in location])
        y = np.array([el[1] for el in location])
    for x, y in zip(x, y):
        dot = plt.Circle((x, y), dot_size, color=color, alpha=0.5)
        ax.add_patch(dot)
    graph = get_graph()
    return graph


# Carry plot function
def carry_map(game_code, player, side):
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.draw()
    plt.title('Ведение мяча: ' + player)
    pitch_height = 80
    pitch_width = 120
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')
    df_carry = df.loc[(df['player_name'] == player) & (df['type_name'] == 'Carry')]
    location = df_carry['location'].tolist()
    carry_end_location = df_carry['carry_end_location'].tolist()
    if side == 'home':
        x1 = np.array([el[0] for el in location])
        y1 = pitch_height - np.array([el[1] for el in location])
        x2 = np.array([el[0] for el in carry_end_location])
        y2 = pitch_height - np.array([el[1] for el in carry_end_location])
    else:
        x1 = pitch_width - np.array([el[0] for el in location])
        y1 = np.array([el[1] for el in location])
        x2 = pitch_width - np.array([el[0] for el in carry_end_location])
        y2 = np.array([el[1] for el in carry_end_location])
    u = x2 - x1
    v = y2 - y1
    color = 'cornflowerblue'

    ax.quiver(x1, y1, u, v, color='#cfcfcf', linewidth=0.003, headlength=4.5, linestyle='dashed')

    graph = get_graph()
    return graph


def shot_map(game_code, player, team, side):
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    df = pd.json_normalize(game, sep='_')
    df_shot = df.loc[(df['player_name'] == player) & (df['type_name'] == 'Shot')]
    location = df_shot['location'].tolist()
    shots = df.loc[(df['player_name'] == player) & (df['type_name'] == 'Shot')]

    # Select the columns
    shots = shots[
        ['team_name', 'player_name', 'minute', 'second', 'location', 'shot_statsbomb_xg', 'shot_outcome_name']]
    shots['x'] = shots.location.apply(lambda x: x[0])
    shots['y'] = shots.location.apply(lambda x: x[1])
    shots = shots.drop('location', axis=1)
    # Divide the dataset based on the outcome
    goals = shots[shots.shot_outcome_name == 'Goal']
    shots = shots[shots.shot_outcome_name != 'Goal']
    shots.head()
    pitch = Pitch(pitch_type='statsbomb', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.grid(figheight=8, title_height=0.08, endnote_space=0, axis=False, title_space=0, grid_height=0.87,
                         endnote_height=0.05)

    ax['title'].text(0.50, 0.40, 'Удары: ' + player, fontsize=22,
                     va='center', ha='center')
    pitch_height = 80
    pitch_width = 120
    color = 'orange' if team == 'Есперион' else 'red'
    if side == 'home':
        x1 = np.array([el[0] for el in location])
        y1 = np.array([el[1] for el in location])
        x2 = np.full((len(x1)), 120)
        y2 = np.full((len(y1)), 40)
        scatter_goals = pitch.scatter(goals.x, goals.y, s=(goals.shot_statsbomb_xg * 900) + 100, c='white',
                                      edgecolors='black', marker='football', ax=ax['pitch'])
        u = x2 - x1
        v = y1 - y2
        # pitch.arrows(x1, y1, u, v, ax=ax['pitch'], color='#c7d5cc', width=0.5, headlength=4.5)
        ax['pitch'].quiver(x1, y1, u, v, color=color, width=0.005, headlength=4.5)
    elif side == 'away':
        x1 = 120 - np.array([el[0] for el in location])
        y1 = 80 - np.array([el[1] for el in location])
        x2 = np.full((len(x1)), 0)
        y2 = np.full((len(y1)), 40)
        scatter_goals = pitch.scatter(120 - goals.x, goals.y, s=(goals.shot_statsbomb_xg * 900) + 100, c='white',
                                      edgecolors='black', marker='football', ax=ax['pitch'])
        u = x2 - x1
        v = y1 - y2
        # pitch.arrows(x1, y1, u, v, ax=ax['pitch'], color='#c7d5cc', width=0.5, headlength=4.5)
        ax['pitch'].quiver(x1, y1, u, v, color=color, width=0.005, headlength=4.5)

    graph = get_graph()
    return graph


def heatmap(game_code):
    parser = Sbopen()
    pitch = Pitch(line_zorder=2)
    bins = (16, 12)  # 16 cells x 12 cells
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)

    all_events_df = []
    cols = ['match_id', 'id', 'type_name', 'sub_type_name', 'player_name',
            'x', 'y', 'end_x', 'end_y', 'outcome_name', 'shot_statsbomb_xg']
    event = pd.json_normalize(game, sep='_')  # get the first dataframe (events) which has index = 0
    event = event.loc[event.type_name.isin(['Carry', 'Shot', 'Pass']), cols].copy()

    # boolean columns for working out probabilities
    event['goal'] = event['outcome_name'] == 'Goal'
    event['shoot'] = event['type_name'] == 'Shot'
    event['move'] = event['type_name'] != 'Shot'
    all_events_df.append(event)

    event = pd.concat(all_events_df)

    shot_probability = pitch.bin_statistic(event['x'], event['y'], values=event['shoot'],
                                           statistic='mean', bins=bins)
    move_probability = pitch.bin_statistic(event['x'], event['y'], values=event['move'],
                                           statistic='mean', bins=bins)
    goal_probability = pitch.bin_statistic(event.loc[event['shoot'], 'x'],
                                           event.loc[event['shoot'], 'y'],
                                           event.loc[event['shoot'], 'goal'],
                                           statistic='mean', bins=bins)
    fig, ax = pitch.draw()
    shot_heatmap = pitch.heatmap(shot_probability, ax=ax)

    fig, ax = pitch.draw()
    move_heatmap = pitch.heatmap(move_probability, ax=ax)

    fig, ax = pitch.draw()
    goal_heatmap = pitch.heatmap(goal_probability, ax=ax)

    # get a dataframe of move events and filter it
    # so the dataframe only contains actions inside the pitch.
    move = event[event['move']].copy()
    bin_start_locations = pitch.bin_statistic(move['x'], move['y'], bins=bins)
    move = move[bin_start_locations['inside']].copy()

    # get the successful moves, which filters out the events that ended outside the pitch
    # or where not successful (null)
    bin_end_locations = pitch.bin_statistic(move['end_x'], move['end_y'], bins=bins)
    move_success = move[(bin_end_locations['inside']) & (move['outcome_name'].isnull())].copy()

    # get a dataframe of the successful moves
    # and the grid cells they started and ended in
    bin_success_start = pitch.bin_statistic(move_success['x'], move_success['y'], bins=bins)
    bin_success_end = pitch.bin_statistic(move_success['end_x'], move_success['end_y'], bins=bins)
    df_bin = pd.DataFrame({'x': bin_success_start['binnumber'][0],
                           'y': bin_success_start['binnumber'][1],
                           'end_x': bin_success_end['binnumber'][0],
                           'end_y': bin_success_end['binnumber'][1]})

    # calculate the bin counts for the successful moves, i.e. the number of moves between grid cells
    bin_counts = df_bin.value_counts().reset_index(name='bin_counts')

    # create the move_transition_matrix of shape (num_y_bins, num_x_bins, num_y_bins, num_x_bins)
    # this is the number of successful moves between grid cells.
    num_y, num_x = shot_probability['statistic'].shape
    move_transition_matrix = np.zeros((num_y, num_x, num_y, num_x))
    move_transition_matrix[bin_counts['y'], bin_counts['x'],
                           bin_counts['end_y'], bin_counts['end_x']] = bin_counts.bin_counts.values

    # and divide by the starting locations for all moves (including unsuccessful)
    # to get the probability of moving the ball successfully between grid cells
    bin_start_locations = pitch.bin_statistic(move['x'], move['y'], bins=bins)
    bin_start_locations = np.expand_dims(bin_start_locations['statistic'], (2, 3))
    move_transition_matrix = np.divide(move_transition_matrix,
                                       bin_start_locations,
                                       out=np.zeros_like(move_transition_matrix),
                                       where=bin_start_locations != 0,
                                       )
    move_transition_matrix = np.nan_to_num(move_transition_matrix)
    shot_probability_matrix = np.nan_to_num(shot_probability['statistic'])
    move_probability_matrix = np.nan_to_num(move_probability['statistic'])
    goal_probability_matrix = np.nan_to_num(goal_probability['statistic'])

    xt = np.multiply(shot_probability_matrix, goal_probability_matrix)
    diff = 1
    iteration = 0
    while np.any(diff > 0.00001):  # iterate until the differences between the old and new xT is small
        xt_copy = xt.copy()  # keep a copy for comparing the differences
        # calculate the new expected threat
        xt = (np.multiply(shot_probability_matrix, goal_probability_matrix) +
              np.multiply(move_probability_matrix,
                          np.multiply(move_transition_matrix, np.expand_dims(xt, axis=(0, 1))).sum(
                              axis=(2, 3)))
              )
        diff = (xt - xt_copy)
        iteration += 1
    print('Number of iterations:', iteration)

    path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()]
    # new bin statistic for plotting xt only
    for_plotting = pitch.bin_statistic(event['x'], event['y'], bins=bins)
    for_plotting['statistic'] = xt
    fig, ax = pitch.draw(figsize=(14, 9.625))
    _ = pitch.heatmap(for_plotting, ax=ax)
    _ = pitch.label_heatmap(for_plotting, ax=ax, str_format='{:.2%}',
                            color='white', fontsize=14, va='center', ha='center',
                            path_effects=path_eff)

    # first get grid start and end cells
    grid_start = pitch.bin_statistic(move_success.x, move_success.y, bins=bins)
    grid_end = pitch.bin_statistic(move_success.end_x, move_success.end_y, bins=bins)

    # then get the xT values from the start and end grid cell
    start_xt = xt[grid_start['binnumber'][1], grid_start['binnumber'][0]]
    end_xt = xt[grid_end['binnumber'][1], grid_end['binnumber'][0]]

    # then calculate the added xT
    added_xt = end_xt - start_xt
    move_success['xt'] = added_xt

    # show players with top 5 total expected threat
    print(move_success.groupby('player_name')['xt'].sum().sort_values(ascending=False).head(5))

    graph = get_graph()
    return graph


def heatmap2(game_code):
    parser = Sbopen()
    pitch = Pitch(line_zorder=2)
    bins = (pitch.dim.positional_x[[0, 3, 4, 5, 6]], pitch.dim.positional_y)
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        game = json.load(f)
    event = pd.json_normalize(game, sep='_')

    all_events_df = []
    set_pieces = ['Throw-in', 'Free Kick', 'Goal Kick', 'Corner', 'Kick Off', 'Penalty']
    cols = ['match_id', 'id', 'type_name', 'sub_type_name', 'player_name',
            'x', 'y', 'end_x', 'end_y', 'outcome_name', 'shot_statsbomb_xg']
    # get carries/ passes/ shots
    event = parser.event(game_code)[0]  # get the first dataframe (events) which has index = 0
    event = event.loc[((event.type_name.isin(['Carry', 'Shot', 'Pass'])) &
                       (~event['sub_type_name'].isin(set_pieces))),  # remove set-piece events
                      cols].copy()

    # boolean columns for working out probabilities
    event['goal'] = event['outcome_name'] == 'Goal'
    event['shoot'] = event['type_name'] == 'Shot'
    event['move'] = event['type_name'] != 'Shot'
    all_events_df.append(event)
    event = pd.concat(all_events_df)

    shot_probability = pitch.bin_statistic(event['x'], event['y'], values=event['shoot'],
                                           statistic='mean', bins=bins)
    move_probability = pitch.bin_statistic(event['x'], event['y'], values=event['move'],
                                           statistic='mean', bins=bins)
    goal_probability = pitch.bin_statistic(event.loc[event['shoot'], 'x'],
                                           event.loc[event['shoot'], 'y'],
                                           event.loc[event['shoot'], 'shot_statsbomb_xg'],
                                           statistic='mean',
                                           bins=bins)

    fig, ax = pitch.draw()
    shot_heatmap = pitch.heatmap(shot_probability, ax=ax)

    fig, ax = pitch.draw()
    move_heatmap = pitch.heatmap(move_probability, ax=ax)

    fig, ax = pitch.draw()
    goal_heatmap = pitch.heatmap(goal_probability, ax=ax)

    # get a dataframe of move events and filter it
    # so the dataframe only contains actions inside the pitch.
    move = event[event['move']].copy()
    bin_start_locations = pitch.bin_statistic(move['x'], move['y'], bins=bins)
    move = move[bin_start_locations['inside']].copy()

    # get the successful moves, which filters out the events that ended outside the pitch
    # or where not successful (null)
    bin_end_locations = pitch.bin_statistic(move['end_x'], move['end_y'], bins=bins)
    move_success = move[(bin_end_locations['inside']) & (move['outcome_name'].isnull())].copy()

    # get a dataframe of the successful moves
    # and the grid cells they started and ended in
    bin_success_start = pitch.bin_statistic(move_success['x'], move_success['y'], bins=bins)
    bin_success_end = pitch.bin_statistic(move_success['end_x'], move_success['end_y'], bins=bins)
    df_bin = pd.DataFrame({'x': bin_success_start['binnumber'][0],
                           'y': bin_success_start['binnumber'][1],
                           'end_x': bin_success_end['binnumber'][0],
                           'end_y': bin_success_end['binnumber'][1]})

    # calculate the bin counts for the successful moves, i.e. the number of moves between grid cells
    bin_counts = df_bin.value_counts().reset_index(name='bin_counts')

    # create the move_transition_matrix of shape (num_y_bins, num_x_bins, num_y_bins, num_x_bins)
    # this is the number of successful moves between grid cells.
    num_y, num_x = shot_probability['statistic'].shape
    move_transition_matrix = np.zeros((num_y, num_x, num_y, num_x))
    move_transition_matrix[bin_counts['y'], bin_counts['x'],
                           bin_counts['end_y'], bin_counts['end_x']] = bin_counts.bin_counts.values

    # and divide by the starting locations for all moves (including unsuccessful)
    # to get the probability of moving the ball successfully between grid cells
    bin_start_locations = pitch.bin_statistic(move['x'], move['y'], bins=bins)
    bin_start_locations = np.expand_dims(bin_start_locations['statistic'], (2, 3))
    move_transition_matrix = np.divide(move_transition_matrix,
                                       bin_start_locations,
                                       out=np.zeros_like(move_transition_matrix),
                                       where=bin_start_locations != 0,
                                       )

    move_transition_matrix = np.nan_to_num(move_transition_matrix)
    shot_probability_matrix = np.nan_to_num(shot_probability['statistic'])
    move_probability_matrix = np.nan_to_num(move_probability['statistic'])
    goal_probability_matrix = np.nan_to_num(goal_probability['statistic'])

    xt = np.multiply(shot_probability_matrix, goal_probability_matrix)
    diff = 1
    iteration = 0
    while np.any(diff > 0.00001):  # iterate until the differences between the old and new xT is small
        xt_copy = xt.copy()  # keep a copy for comparing the differences
        # calculate the new expected threat
        xt = (np.multiply(shot_probability_matrix, goal_probability_matrix) +
              np.multiply(move_probability_matrix,
                          np.multiply(move_transition_matrix, np.expand_dims(xt, axis=(0, 1))).sum(
                              axis=(2, 3)))
              )
        diff = (xt - xt_copy)
        iteration += 1
    print('Number of iterations:', iteration)

    path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()]
    # new bin statistic for plotting xt only
    for_plotting = pitch.bin_statistic(event['x'], event['y'], bins=bins)
    for_plotting['statistic'] = xt
    fig, ax = pitch.draw(figsize=(14, 9.625))
    _ = pitch.heatmap(for_plotting, ax=ax)
    _ = pitch.label_heatmap(for_plotting, ax=ax, str_format='{:.2%}',
                            color='white', fontsize=14, va='center', ha='center',
                            path_effects=path_eff)
    # sphinx_gallery_thumbnail_path = 'gallery/tutorials/images/sphx_glr_plot_xt_improvements_004'

    # first get grid start and end cells
    grid_start = pitch.bin_statistic(move_success.x, move_success.y, bins=bins)
    grid_end = pitch.bin_statistic(move_success.end_x, move_success.end_y, bins=bins)

    # then get the xT values from the start and end grid cell
    start_xt = xt[grid_start['binnumber'][1], grid_start['binnumber'][0]]
    end_xt = xt[grid_end['binnumber'][1], grid_end['binnumber'][0]]

    # then calculate the added xT
    added_xt = end_xt - start_xt
    move_success['xt'] = added_xt

    # show players with top 5 total expected threat
    print(move_success.groupby('player_name')['xt'].sum().sort_values(ascending=False).head(5))
    graph = get_graph()
    return graph


def open_json(game_code):
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        data = json.load(f)
        return data


def team_shots(game_code, team, team_color, half):
    data = open_json(game_code)
    df = pd.json_normalize(data, sep='_')
    df.head()

    if half == 'Первый тайм':
        chosen_half = df.loc[:1808, :]
    elif half == 'Второй тайм':
        chosen_half = df.loc[1809:, :]
    elif half == '90 минут':
        chosen_half = df.loc[:, :]

    VerticalPitch(pitch_type='statsbomb', half=True)
    # Retrieve rows that record shots
    shots = chosen_half[chosen_half.type_name == 'Shot']
    # Filter the data that record AC Milan
    shots = shots[shots.team_name == team]
    # Select the columns
    shots = shots[
        ['team_name', 'player_name', 'minute', 'second', 'location', 'shot_statsbomb_xg', 'shot_outcome_name']]
    # Because the location data is on list format (ex: [100, 80]), we extract the x and y coordinate using apply method.
    shots['x'] = shots.location.apply(lambda x: x[0])
    shots['y'] = shots.location.apply(lambda x: x[1])
    shots = shots.drop('location', axis=1)
    total_xg = shots["shot_statsbomb_xg"].sum().round(2)
    # Divide the dataset based on the outcome
    goals = shots[shots.shot_outcome_name == 'Goal']
    shots = shots[shots.shot_outcome_name != 'Goal']
    shots.head()
    total_shots = shots.shape[0]
    total_goals = goals.shape[0]

    pitch = VerticalPitch(pitch_type='statsbomb', half=True, goal_type='box', goal_alpha=0.8, pitch_color='#22312b',
                          line_color='#c7d5cc')
    fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0, axis=False, title_space=0, grid_height=0.82,
                          endnote_height=0.05)
    fig.set_facecolor("#22312b")

    if team_color == 'red':
        scatter_shots = pitch.scatter(shots.x, shots.y, s=(shots.shot_statsbomb_xg * 900) + 100, c='red',
                                      edgecolors='black', marker='o', ax=axs['pitch'])
        scatter_goals = pitch.scatter(goals.x, goals.y, s=(goals.shot_statsbomb_xg * 900) + 100, c='white',
                                      edgecolors='black', marker='football', ax=axs['pitch'])
    else:
        scatter_shots = pitch.scatter(shots.x, shots.y, s=(shots.shot_statsbomb_xg * 900) + 100, c='purple',
                                      edgecolors='black', marker='o', ax=axs['pitch'])
        scatter_goals = pitch.scatter(goals.x, goals.y, s=(goals.shot_statsbomb_xg * 900) + 100, c='white',
                                      edgecolors='black', marker='football', ax=axs['pitch'])

    pitch.arrows(70, 5, 100, 5, ax=axs['pitch'], color='#c7d5cc')
    axs['endnote'].text(0.85, 0.5, '[YOUR NAME]', color='#c7d5cc', va='center', ha='center', fontsize=15)
    axs['title'].text(0.5, 0.7, 'Удары: ' + team, color='#c7d5cc', va='center', ha='center', fontsize=30)
    axs['title'].text(0.5, 0.25, half, color='#c7d5cc', va='center', ha='center', fontsize=18)

    basic_info_txt = "Удары: " + str(total_shots) + " | Голы: " + str(total_goals) + " | Ожидаемые голы: " + str(
        total_xg)
    fig.text(0.5, 0.3, basic_info_txt,
             size=20,
             ha="center", color="white")

    graph = get_graph()
    return graph


def team_pressure(game_code, team, team_color, half):
    data = open_json(game_code)
    df = pd.json_normalize(data, sep='_')
    df.head()

    if half == 'Первый тайм':
        chosen_half = df.loc[:1808, :]
    elif half == 'Второй тайм':
        chosen_half = df.loc[1809:3551, :]
    elif half == '90 минут':
        chosen_half = df.loc[:, :]

    pressure = chosen_half[df.type_name == 'Pressure']
    pressure = pressure[['team_name', 'player_name', 'location']]
    pressure = pressure[pressure.team_name == team]
    pressure['x'] = pressure.location.apply(lambda x: x[0])
    pressure['y'] = pressure.location.apply(lambda x: x[1])
    pressure = pressure.drop('location', axis=1)
    pressure.head()

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
    fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0, axis=False, title_space=0, grid_height=0.82,
                          endnote_height=0.05)
    fig.set_facecolor('#22312b')
    bin_statistic = pitch.bin_statistic(pressure.x, pressure.y, statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    LinearSegmentedColormap.from_list("El Greco Violet - 100 colors",
                                      ['#3b3154', '#8e78a0'], N=100)
    if team_color == 'blue':
        pcm = pitch.heatmap(bin_statistic, ax=axs['pitch'], edgecolors='#20143f', cmap=cmr.voltage)
    else:
        pcm = pitch.heatmap(bin_statistic, ax=axs['pitch'], cmap='hot', edgecolors='#22312b')
    cbar = fig.colorbar(pcm, ax=axs['pitch'], shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
    axs['endnote'].text(0.8, 0.5, '[YOUR NAME]', color='#c7d5cc', va='center', ha='center', fontsize=10)
    axs['endnote'].text(0.4, 0.95, 'Attacking Direction', va='center', ha='center', color='#c7d5cc', fontsize=12)
    axs['endnote'].arrow(0.3, 0.6, 0.2, 0, head_width=0.2, head_length=0.025, ec='w', fc='w')
    axs['endnote'].set_xlim(0, 1)
    axs['endnote'].set_ylim(0, 1)
    axs['title'].text(0.5, 0.7, 'Тепловая карта прессинга: ' + team, color='#c7d5cc', va='center', ha='center',
                      fontsize=30)
    axs['title'].text(0.5, 0.25, half, color='#c7d5cc', va='center', ha='center', fontsize=18)

    graph = get_graph()
    return graph


def create_xticks(period, minute):
    if period == 1:
        return str(minute)
    else:
        return str(minute) + "'"


def xg(game_code):
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        data = json.load(f)
    df = pd.json_normalize(data, sep='_')
    df.head()

    shots = df[df.type_name == 'Shot']
    shots.head()
    shots = shots[['timestamp', 'period', 'minute', 'team_name', 'shot_outcome_name', 'shot_statsbomb_xg']]
    shots['time'] = shots.apply(lambda x: create_xticks(x['period'], x['minute']), axis=1)
    shots.head()
    shots_tidy = shots.groupby(['team_name', 'period', 'minute', 'time', 'shot_outcome_name']).sum().groupby(
        level=0).cumsum().reset_index()
    shots_tidy.head()
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.lineplot('minute', 'shot_statsbomb_xg', hue='team_name', data=shots_tidy, ax=ax)

    ax.title.set_text('Прогресс ожидаемых голов')
    legend = ax.legend()
    ax.set(xlabel='Минута')
    ax.set(ylabel='Ожидаемые голы')

    loc_x = plticker.MultipleLocator(base=5.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc_x)

    plt.xticks()
    graph = get_graph()
    return graph
    # sphinx_gallery_thumbnail_path = 'gallery/pitch_plots/images/sphx_glr_plot_heatmap_positional_002.png'


def pass2(game_code, team1, team2, color):
    filename = './static/games/' + game_code + '.json'
    with open(filename, 'r', errors="ignore", encoding='utf-8') as f:
        data = json.load(f)
        df = pd.json_normalize(data, sep="_")

        ## Get the necessary dataframe for both the teams ready
        team_pass_btn, team_avg_psn = build_net(df, team1)

        ##Draw the passing networks for both teams
        draw_network(team_pass_btn, team_avg_psn, team1, team2, color)

    graph = get_graph()
    return graph


def build_net(df, team):
    ##Filter the df with only passes completed by the given team
    team_passes = df[(df['type_name'] == 'Pass') & (df['team_name'] == team)][
        ['location', 'pass_end_location', 'player_name', 'pass_recipient_name', 'minute']]

    ##Find 1st substitution

    minute = df[(df['type_name'] == 'Substitution') & (df['team_name'] == team)]['minute'].iloc[0]
    ##Consider passes till only 1st sub
    team_passes = team_passes[team_passes['minute'] < minute]

    team_passes[['start_x', 'start_y']] = pd.DataFrame(team_passes.location.tolist(), index=team_passes.index)
    team_passes[['end_x', 'end_y']] = pd.DataFrame(team_passes.pass_end_location.tolist(), index=team_passes.index)
    team_passes.drop(['location', 'pass_end_location'], axis=1, inplace=True)

    ##Calculate average player positions based on the pass starting points
    average_positions = team_passes.groupby('player_name').mean()
    average_positions = average_positions.reset_index()
    average_positions = average_positions.drop(['end_x', 'end_y'], axis=1)

    jerseys = pd.DataFrame(columns=['player_name'])
    jerseys['player_name'] = average_positions['player_name']

    average_positions = average_positions.merge(jerseys, on='player_name')

    ##Create a dataframe for pair-wise passing
    passes_between = team_passes.groupby(['player_name', 'pass_recipient_name']).agg(
        {'start_y': ['mean', 'count']}).reset_index()
    ##Get the number of passes made
    passes_count = passes_between['start_y', 'count']

    passes_between['count'] = passes_count
    passes_between = passes_between.drop('start_y', axis=1)
    passes_between.columns = [''.join(col) for col in passes_between.columns]

    ##Putting all the pieces together
    passes_between = passes_between.merge(average_positions, left_on='player_name', right_on='player_name')

    passes_between = passes_between.merge(average_positions, left_on='pass_recipient_name', right_on='player_name')

    passes_between.rename({'player_name_x': 'player_name'}, axis='columns', inplace=True)
    passes_between.rename({'start_x_x': 'start_x'}, axis='columns', inplace=True)
    passes_between.rename({'start_y_x': 'start_y'}, axis='columns', inplace=True)
    passes_between.rename({'start_x_y': 'end_x'}, axis='columns', inplace=True)
    passes_between.rename({'start_y_y': 'end_y'}, axis='columns', inplace=True)
    passes_between.rename({'number_x': 'start_number'}, axis='columns', inplace=True)
    passes_between.rename({'number_y': 'end_number'}, axis='columns', inplace=True)
    passes_between = passes_between.drop(['minute_x', 'player_name_y', 'minute_y'], axis=1)

    return passes_between, average_positions


##Function to draw the network
def draw_network(team_passes, team_avg, team, team2, color):
    ##Passing Network for the 1st team

    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='#22312b', line_color='white')

    fig, ax1 = pitch.draw()
    fig.set_facecolor("#22312b")
    pitch.scatter(team_avg['start_x'], team_avg['start_y'], ax=ax1, s=100, color=color)
    ##Loop through the average position df to plot the average location of each player

    ##We'll be plotting only passes between pairs that happened more than twice
    pass_between = team_passes[team_passes['count'] > 2]
    pitch.arrows(pass_between['start_x'], pass_between['start_y'], pass_between['end_x'], pass_between['end_y'], ax=ax1,
                 color='white',
                 alpha=0.4, width=1.8, headwidth=4, headlength=4)

    for p in range(len(team_avg)):
        pitch.annotate(text=team_avg.iloc[p]['player_name'],
                       xy=(team_avg.iloc[p]['start_x'] - 5, team_avg.iloc[p]['start_y'] - 3),
                       ax=ax1, c='white', va='center', ha='center', size=5, fontweight='bold',
                       fontfamily="Century Gothic", backgroundcolor='black')

    fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0,
                          # Turn off the endnote/title axis. I usually do this after
                          # I am happy with the chart layout and text placement
                          axis=False,
                          title_space=0, grid_height=0.82, endnote_height=0.05)
    fig.set_facecolor("#22312b")
    pitch.scatter(team_avg['start_x'], team_avg['start_y'], ax=axs['pitch'], s=800, color=color)
    ##Loop through the average position df to plot the average location of each player

    ##We'll be plotting only passes between pairs that happened more than twice
    pass_between = team_passes[team_passes['count'] > 2]
    pitch.arrows(pass_between['start_x'], pass_between['start_y'], pass_between['end_x'], pass_between['end_y'],
                 ax=axs['pitch'],
                 color='white',
                 alpha=0.5, width=2, headwidth=4, headlength=4)

    for p in range(len(team_avg)):
        pitch.annotate(text=team_avg.iloc[p]['player_name'],
                       xy=(team_avg.iloc[p]['start_x'] - 5, team_avg.iloc[p]['start_y'] - 3),
                       ax=axs['pitch'], c='white', va='center', ha='center', size=10, fontweight='bold',
                       fontfamily="Century Gothic", backgroundcolor='black')

    ##Describe
    TITLE_TEXT = f'{team} | Карта пасов'
    axs['title'].text(0.5, 0.7, TITLE_TEXT, color='#c7d5cc',
                      va='center', ha='center', fontsize=30)
    axs['title'].text(0.5, 0.25, '90 минут vs ' + team2, color='#c7d5cc',
                      va='center', ha='center', fontsize=18)


def convex_hull(game_code, player, side):
    data = open_json(game_code)
    df = pd.json_normalize(data, sep='_')

    # Filter passes by Jodie Taylor
    df = df[(df.player_name == player) & (df.type_name == 'Pass')].copy()
    location = df['location'].tolist()
    ##############################################################################
    # Plotting

    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80, pitch_color='grass', line_color='white',
                  stripe=True)
    fig, ax = pitch.draw()
    if side == 'home':
        hull = pitch.convexhull(np.array([el[0] for el in location]), 80 - np.array([el[1] for el in location]))
        poly = pitch.polygon(hull, ax=ax, edgecolor='cornflowerblue', facecolor='cornflowerblue', alpha=0.3)
        scatter = pitch.scatter(np.array([el[0] for el in location]), 80 - np.array([el[1] for el in location]), ax=ax,
                                edgecolor='black', facecolor='cornflowerblue')
    elif side == 'away':
        hull = pitch.convexhull(120 - np.array([el[0] for el in location]), np.array([el[1] for el in location]))
        poly = pitch.polygon(hull, ax=ax, edgecolor='cornflowerblue', facecolor='cornflowerblue', alpha=0.3)
        scatter = pitch.scatter(120 - np.array([el[0] for el in location]), np.array([el[1] for el in location]), ax=ax,
                                edgecolor='black', facecolor='cornflowerblue')
    # if you are not using a Jupyter notebook this is necessary to show the plot
    plt.title('Зона передач')  # <-- change to title

    graph = get_graph()
    return graph
# def check():
# ax=pass_map()
# fig.set_size_inches(15, 10)
# graph = get_graph()
#  return graph
