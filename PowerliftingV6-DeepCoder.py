import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import json
import itertools

# Initialize session state
if 'training_max' not in st.session_state:
    st.session_state.training_max = {}
if 'current_cycle' not in st.session_state:
    st.session_state.current_cycle = 1
if 'current_week' not in st.session_state:
    st.session_state.current_week = 1
if 'set_number' not in st.session_state:
    st.session_state.set_number = 1
if 'available_weights' not in st.session_state:
    st.session_state.available_weights = {1.25: 2, 2.5: 2, 5: 2, 10: 2, 20: 2}  # in kg: count
if 'lift_data' not in st.session_state:
    st.session_state.lift_data = pd.DataFrame(columns=['Date', 'Lift', 'Set', 'Weight', 'Reps', 'Estimated_1RM', 'Cycle', 'Week'])
if 'completed_cycles' not in st.session_state:
    st.session_state.completed_cycles = []

# Helper functions
def estimate_1rm(weight, reps):
    return round(weight * (1 + 0.0333 * reps), 2)

def get_week_lifts(training_max, week):
    percentages = {
        1: [(0.65, 5), (0.75, 5), (0.85, "5+")],
        2: [(0.70, 3), (0.80, 3), (0.90, "3+")],
        3: [(0.75, 5), (0.85, 3), (0.95, "1+")],
        4: [(0.40, 5), (0.50, 5), (0.60, 5)]
    }
    return [(round_weight(training_max * p), r) for p, r in percentages[week]]

def round_weight(weight):
    return round(weight / 2.5) * 2.5

# Data management functions
def load_data():
    try:
        return pd.read_json('workout_data.json')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Date', 'Lift', 'Set', 'Weight', 'Reps', 'Estimated_1RM', 'Cycle', 'Week'])

def save_data(df):
    df.to_json('workout_data.json', date_format='iso', orient='records')

def load_training_max():
    try:
        with open('training_max.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_training_max(tm_dict):
    with open('training_max.json', 'w') as f:
        json.dump(tm_dict, f)

def save_available_weights(weights_dict):
    with open('available_weights.json', 'w') as f:
        json.dump(weights_dict, f)

def load_available_weights():
    try:
        with open('available_weights.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {1.25: 2, 2.5: 2, 5: 2, 10: 2, 20: 2}

# Initialize data
st.session_state.lift_data = load_data()
st.session_state.training_max = load_training_max()
st.session_state.available_weights = load_available_weights()

# Main tracking functions
def record_lift(date, lift, set_number, weight, reps, cycle, week):
    estimated_1rm = estimate_1rm(weight, reps)
    new_row = pd.DataFrame({
        'Date': [date], 'Lift': [lift], 'Set': [set_number],
        'Weight': [weight], 'Reps': [reps], 'Estimated_1RM': [estimated_1rm],
        'Cycle': [cycle], 'Week': [week]
    })
    st.session_state.lift_data = pd.concat([st.session_state.lift_data, new_row], ignore_index=True)
    save_data(st.session_state.lift_data)

def update_training_max():
    for lift in st.session_state.training_max:
        if lift in ['Squat', 'Deadlift']:
            st.session_state.training_max[lift] = round_weight(st.session_state.training_max[lift] + 5)
        else:
            st.session_state.training_max[lift] = round_weight(st.session_state.training_max[lift] + 2.5)
    save_training_max(st.session_state.training_max)

# Streamlit UI
st.set_page_config(page_title="5/3/1 Workout Tracker", layout="wide")

st.title('5/3/1 Workout Tracker')

# Sidebar for navigation
page = st.sidebar.radio('Navigate', ['Record Lift', 'View Progress', 'Manage Training Max', 'Manage Weights', 'View Previous Cycles'])

if page == 'Record Lift':
    st.header('Record Lift')
    
    st.write(f"Current Cycle: {st.session_state.current_cycle}, Week: {st.session_state.current_week}")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input('Date', datetime.now())
    with col2:
        lift = st.selectbox('Lift', ['Squat', 'Deadlift', 'Overhead Press'])
    
    if lift in st.session_state.training_max:
        training_max = st.session_state.training_max[lift]
        week_lifts = get_week_lifts(training_max, st.session_state.current_week)
        
        st.subheader("Today's Lifts")
        completed_sets = 0
        for i, (weight, reps) in enumerate(week_lifts):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"Set {i+1}: {weight:.1f} kg x {reps} reps")
            with col2:
                weight_input = st.number_input(f'Weight for Set {i+1}', value=float(weight), step=2.5, key=f'weight_{i+1}')
            with col3:
                reps_input = st.number_input(f'Reps for Set {i+1}', value=reps if isinstance(reps, int) else 5, min_value=1, key=f'reps_{i+1}')
            with col4:
                set_completed = st.checkbox(f'Set {i+1} Completed', key=f'completed_{i+1}')
            
            if set_completed:
                record_lift(date, lift, i+1, weight_input, reps_input, st.session_state.current_cycle, st.session_state.current_week)
                st.success(f'Set {i+1} recorded successfully!')
                completed_sets += 1

        if completed_sets == len(week_lifts):
            col1, col2 = st.columns(2)
            with col1:
                if st.button('End Week', key='end_week'):
                    if st.session_state.current_week < 4:
                        st.session_state.current_week += 1
                    else:
                        st.session_state.current_week = 1
                        st.session_state.completed_cycles.append(st.session_state.current_cycle)
                        st.session_state.current_cycle += 1
                    st.session_state.set_number = 1
                    st.success(f'Moving to Week {st.session_state.current_week}, Cycle {st.session_state.current_cycle}')
            with col2:
                if st.button('Start New Cycle', key='new_cycle'):
                    update_training_max()
                    st.session_state.current_week = 1
                    st.session_state.completed_cycles.append(st.session_state.current_cycle)
                    st.session_state.current_cycle += 1
                    st.session_state.set_number = 1
                    st.success(f'New cycle started: Cycle {st.session_state.current_cycle}. Training maxes updated.')

    else:
        st.warning('Please set your Training Max for this lift in the Manage Training Max section.')

elif page == 'View Progress':
    st.header('View Progress')
    
    lift = st.selectbox('Select Lift', ['Squat', 'Deadlift', 'Overhead Press'])
    
    if not st.session_state.lift_data.empty and lift in st.session_state.lift_data['Lift'].values:
        lift_data = st.session_state.lift_data[st.session_state.lift_data['Lift'] == lift]
        max_weight_data = lift_data.groupby('Date').agg({'Weight': 'max', 'Estimated_1RM': 'max'}).reset_index()
        
        chart = alt.Chart(max_weight_data).mark_line(point=True).encode(
            x='Date:T',
            y='Weight:Q',
            tooltip=['Date', 'Weight']
        ).properties(
            title=f'{lift} Max Weight Progress'
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

        st.subheader('Latest Estimated 1RM')
        latest_1rm = lift_data.loc[lift_data['Estimated_1RM'].idxmax()]
        st.write(f"Date: {latest_1rm['Date']}")
        st.write(f"Weight: {latest_1rm['Weight']} kg")
        st.write(f"Reps: {latest_1rm['Reps']}")
        st.write(f"Estimated 1RM: {latest_1rm['Estimated_1RM']} kg")
    else:
        st.info(f"No data available for {lift}. Start lifting to see your progress!")

elif page == 'Manage Training Max':
    st.header('Manage Training Max')
    
    lifts = ['Squat', 'Deadlift', 'Overhead Press']
    
    for lift in lifts:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f'{lift}')
        with col2:
            current_tm = st.session_state.training_max.get(lift, 0)
            new_tm = st.number_input(f'{lift} TM (kg)', value=float(current_tm), step=2.5, key=f'tm_{lift}')
            if st.button(f'Update {lift} TM', key=f'update_{lift}'):
                st.session_state.training_max[lift] = round_weight(new_tm)
                save_training_max(st.session_state.training_max)
                st.success(f'{lift} Training Max updated to {round_weight(new_tm)}kg')

elif page == 'Manage Weights':
    st.header('Manage Available Weights')
    
    st.write("Enter the weights you have available and how many of each (in kg).")
    
    new_weight = st.number_input('Add new weight (kg)', value=0.0, step=0.5, min_value=0.0)
    if st.button('Add New Weight'):
        if new_weight > 0 and new_weight not in st.session_state.available_weights:
            st.session_state.available_weights[new_weight] = 0
            st.success(f'Added new weight: {new_weight}kg')
        else:
            st.warning('Please enter a valid weight that is not already in the list.')
    
    weights_to_remove = []
    for weight in sorted(st.session_state.available_weights.keys()):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f'{weight}kg')
        with col2:
            count = st.number_input(f'Number of {weight}kg weights', value=st.session_state.available_weights[weight], min_value=0, key=f'weight_{weight}')
            st.session_state.available_weights[weight] = count
        with col3:
            if st.button(f'Remove {weight}kg', key=f'remove_{weight}'):
                weights_to_remove.append(weight)
    
    for weight in weights_to_remove:
        del st.session_state.available_weights[weight]
        st.success(f'Removed {weight}kg from available weights.')
    
    if st.button('Save Changes'):
        save_available_weights(st.session_state.available_weights)
        st.success('Available weights updated and saved successfully!')
    
    st.subheader('Current Available Weights')
    for weight, count in sorted(st.session_state.available_weights.items()):
        st.write(f"{count} x {weight}kg")
    
    st.subheader('Possible Weight Increments')
    increments = set()
    for r in range(1, sum(st.session_state.available_weights.values()) + 1):
        for combo in itertools.combinations_with_replacement(
            [w for w, c in st.session_state.available_weights.items() for _ in range(c)], r):
            increments.add(round_weight(sum(combo)))
    st.write(', '.join(map(str, sorted([inc for inc in increments if inc >= 2.5]))) + ' kg')

elif page == 'View Previous Cycles':
    st.header('View Previous Cycles')
    
    if st.session_state.completed_cycles:
        for cycle in st.session_state.completed_cycles:
            with st.expander(f'Cycle {cycle}'):
                cycle_data = st.session_state.lift_data[st.session_state.lift_data['Cycle'] == cycle]
                for lift in ['Squat', 'Deadlift', 'Overhead Press']:
                    lift_data = cycle_data[cycle_data['Lift'] == lift]
                    if not lift_data.empty:
                        st.subheader(f"{lift}:")
                        st.dataframe(lift_data[['Week', 'Set', 'Weight', 'Reps']])
    else:
        st.info("No completed cycles yet. Complete a cycle to see it here!")

# Deload reminder
if st.session_state.current_week == 4:
    st.sidebar.warning("This is a deload week. Use lighter weights for recovery!")

st.sidebar.markdown("---")
st.sidebar.info("5/3/1 Workout Tracker v2.4")