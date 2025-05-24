import datetime
from calendar_client import get_upcoming_events
from ride_booking_client import book_ride

class VoiceAgent:
  """
  A voice-activated agent that can handle calendar events and ride booking.
  """
  def __init__(self):
    """
    Initializes the VoiceAgent.
    """
    pass

  def _parse_destination_and_time(self, command_text_lower: str):
    """
    Helper function to parse destination and time from command text.
    Assumes command_text_lower is already lowercased.
    """
    destination = "the usual place"
    time_str = "ASAP"

    # Try to find "to"
    to_index = command_text_lower.find("to ")
    if to_index != -1:
        # Potential destination starts after "to "
        potential_dest_start = to_index + 3
        
        # Look for "at" or "for" to delimit destination
        at_index = command_text_lower.find(" at ", potential_dest_start)
        for_index = command_text_lower.find(" for ", potential_dest_start)

        time_keyword_index = -1
        time_keyword = ""

        if at_index != -1 and (for_index == -1 or at_index < for_index):
            time_keyword_index = at_index
            time_keyword = " at "
        elif for_index != -1 and (at_index == -1 or for_index < at_index):
            time_keyword_index = for_index
            time_keyword = " for "
        
        if time_keyword_index != -1:
            destination = command_text_lower[potential_dest_start:time_keyword_index].strip()
            time_str = command_text_lower[time_keyword_index + len(time_keyword):].strip()
        else:
            # No time keyword found after "to", so rest of string is destination
            destination = command_text_lower[potential_dest_start:].strip()
    else: # "to " not found, check if "ride to" was the trigger
        if "ride to " in command_text_lower: # This case might be redundant if "to " is always present with "ride to "
            pass # Handled by the general "to " logic. If not, specific logic for "ride to X" without further "at/for"


    # A simple fallback if destination parsing yields empty string
    if not destination:
        destination = "an unspecified location"
        
    # If time string implies "now" or is vague, make it more concrete for the mock
    # For this mock, we'll keep it simple. If it's "ASAP", "now", or similar,
    # we could convert it to an actual ISO time, but the ride_booking_client
    # currently just takes the string.
    if time_str.lower() in ["asap", "now", ""]:
        time_str = datetime.datetime.now().isoformat() # Default to now for mock
    
    return destination, time_str


  def process_command(self, command_text: str) -> str:
    """
    Processes a user's command and returns an appropriate response.

    Args:
      command_text: The user's command as a string.

    Returns:
      A string containing the agent's response.
    """
    command_text_lower = command_text.lower()

    if "calendar" in command_text_lower or "events" in command_text_lower:
      events = get_upcoming_events()
      if not events:
        return "You have no upcoming events."
      
      response_parts = ["Here are your upcoming events:"]
      for event in events:
        # Ensure start_time is present and correctly formatted
        start_time_str = event.get('start_time', 'Unknown time')
        try:
            # Attempt to parse and reformat if it's a full datetime object
            # For this mock, assume it's already a string as per calendar_client
            pass 
        except Exception:
            pass # Keep original string if parsing/formatting fails
        response_parts.append(f"- {event.get('summary', 'No summary')} at {start_time_str}.")
      return " ".join(response_parts)

    elif "book a ride" in command_text_lower or "ride to" in command_text_lower:
      destination, time_str = self._parse_destination_and_time(command_text_lower)
      
      # For this mock, if time_str is something like "tomorrow at 3 pm",
      # it won't be a valid ISO time for the ride_booking_client.
      # The ride_booking_client expects an ISO string.
      # We'll just pass it along for now as per requirements,
      # but a real implementation would need robust time parsing.
      # If time_str is "ASAP", we convert it to now()
      if time_str == "ASAP":
          time_str = datetime.datetime.now().isoformat()
      # A more robust solution would parse human-readable times here.
      # For now, we'll assume the user might provide an ISO time or a simple phrase.

      booking_response = book_ride(destination, time_str)
      return booking_response.get('message', "Could not process the ride booking request.")

    else:
      return "Sorry, I didn't understand that. I can help with calendar events and booking rides."

if __name__ == "__main__":
  agent = VoiceAgent()

  # Test calendar command
  calendar_command = "What are my events for today?"
  calendar_response = agent.process_command(calendar_command)
  print(f"User: {calendar_command}")
  print(f"Agent: {calendar_response}\n")

  # Test ride booking command (more specific time)
  # ride_command_specific = "Can you book a ride to the library for tomorrow at 3 PM?"
  # For the current mock, time needs to be ISO or "ASAP"
  # Let's simulate a command that might be parsed into something usable
  # ride_command_specific = "book a ride to the library for " + (datetime.datetime.now() + datetime.timedelta(days=1, hours=15)).isoformat()
  ride_command_specific = "book a ride to the central library for tomorrow at 3pm" # Test parsing
  ride_response_specific = agent.process_command(ride_command_specific)
  print(f"User: {ride_command_specific}")
  print(f"Agent: {ride_response_specific}\n")
  
  # Test ride booking command with "at"
  ride_command_at = "Can you book a ride to the airport at 6 AM?"
  ride_response_at = agent.process_command(ride_command_at)
  print(f"User: {ride_command_at}")
  print(f"Agent: {ride_response_at}\n")

  # Test ride booking command with only destination
  ride_command_dest_only = "I need a ride to the coffee shop"
  ride_response_dest_only = agent.process_command(ride_command_dest_only)
  print(f"User: {ride_command_dest_only}")
  print(f"Agent: {ride_response_dest_only}\n")
  
  # Test ride booking with "to" but no explicit time keyword
  ride_command_to_no_time_keyword = "Book a ride to my home"
  ride_response_to_no_time_keyword = agent.process_command(ride_command_to_no_time_keyword)
  print(f"User: {ride_command_to_no_time_keyword}")
  print(f"Agent: {ride_response_to_no_time_keyword}\n")


  # Test unrecognized command
  unrecognized_command = "What's the weather like?"
  unrecognized_response = agent.process_command(unrecognized_command)
  print(f"User: {unrecognized_command}")
  print(f"Agent: {unrecognized_response}\n")

  # Test calendar with no events (assuming get_upcoming_events can return empty)
  # This requires modifying calendar_client or ensuring it can return an empty list.
  # For now, we assume it returns some mock events.
  # To test "no upcoming events", one would need to ensure get_upcoming_events() returns []
  # For example, by calling it with a very large days_ahead or specific conditions.
  # This test is more about the agent's handling of an empty list.
  
  # Test "ride to" without any other details
  ride_to_nothing = "ride to"
  ride_to_nothing_response = agent.process_command(ride_to_nothing)
  print(f"User: {ride_to_nothing}")
  print(f"Agent: {ride_to_nothing_response}\n")

  # Test "book a ride" without any other details
  book_a_ride_nothing = "book a ride"
  book_a_ride_nothing_response = agent.process_command(book_a_ride_nothing)
  print(f"User: {book_a_ride_nothing}")
  print(f"Agent: {book_a_ride_nothing_response}\n")
