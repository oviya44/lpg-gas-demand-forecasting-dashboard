def calculate_excess_shortage(df_current):
    """
    df_current: today's or latest data with 'stock' and predicted 'demand_next'
    """
    df = df_current.copy()
    df['shortage'] = (df['demand_next'] - df['stock']).clip(lower=0)
    df['excess']  = (df['stock'] - df['demand_next']).clip(lower=0)
    return df

def suggest_redistribution(df_with_short_excess):
    suggestions = []
    excess_rows = df_with_short_excess[df_with_short_excess['excess'] > 0].sort_values('excess', ascending=False)
    shortage_rows = df_with_short_excess[df_with_short_excess['shortage'] > 0].sort_values('shortage', ascending=False)
    
    for _, ex in excess_rows.iterrows():
        if ex['excess'] <= 0:
            continue
        for _, sh in shortage_rows.iterrows():
            if sh['shortage'] <= 0 or ex['excess'] <= 0:
                continue
            move = min(ex['excess'], sh['shortage'])
            if move > 50:  # only suggest meaningful transfers
                suggestions.append({
                    'from_district': ex['district'],
                    'from_branch': ex['branch'],
                    'to_district': sh['district'],
                    'to_branch': sh['branch'],
                    'move_cylinders': int(move)
                })
                ex['excess'] -= move
                sh['shortage'] -= move
    
    return suggestions