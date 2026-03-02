def apply_rfc_rule(row):
    if (
        row['High_Priority'] == 1 or
        row['Impact'] == 'High' or
        row['Priority'] == 'High' or
        row['No_of_Related_Incidents'] > 2 or
        row['No_of_Related_Changes'] == 0
    ):
        return 1
    return 0
