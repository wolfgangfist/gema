import datetime

def get_upcoming_events(days_ahead=1):
  """
  Retrieves a list of upcoming calendar events.

  Args:
    days_ahead: An integer representing how many days ahead to fetch events for.
                Defaults to 1.

  Returns:
    A list of dictionaries, where each dictionary represents a calendar event.
    Each event dictionary has the following keys:
      'summary': (str) The event summary.
      'start_time': (str) The event start time in ISO format.
      'end_time': (str) The event end time in ISO format.
  """
  events = []
  
  # Get the current date and time
  now = datetime.datetime.now()

  # Sample Event 1
  event1_start = now + datetime.timedelta(hours=1)
  event1_end = event1_start + datetime.timedelta(hours=1)
  events.append({
      'summary': "Team Meeting",
      'start_time': event1_start.isoformat(),
      'end_time': event1_end.isoformat()
  })

  # Sample Event 2
  event2_start = now + datetime.timedelta(days=days_ahead, hours=3)
  event2_end = event2_start + datetime.timedelta(minutes=90)
  events.append({
      'summary': "Lunch with Alex",
      'start_time': event2_start.isoformat(),
      'end_time': event2_end.isoformat()
  })
  
  # Sample Event 3
  event3_start = now + datetime.timedelta(days=days_ahead, hours=5)
  event3_end = event3_start + datetime.timedelta(hours=2)
  events.append({
      'summary': "Project Deadline",
      'start_time': event3_start.isoformat(),
      'end_time': event3_end.isoformat()
  })

  return events

if __name__ == "__main__":
  upcoming_events = get_upcoming_events()
  print("Upcoming Events (next day):")
  for event in upcoming_events:
    print(f"  Summary: {event['summary']}")
    print(f"  Start: {event['start_time']}")
    print(f"  End: {event['end_time']}")
    print("-" * 20)

  upcoming_events_3_days = get_upcoming_events(days_ahead=3)
  print("\nUpcoming Events (next 3 days):")
  for event in upcoming_events_3_days:
    print(f"  Summary: {event['summary']}")
    print(f"  Start: {event['start_time']}")
    print(f"  End: {event['end_time']}")
    print("-" * 20)
