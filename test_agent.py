import unittest
from unittest.mock import patch, MagicMock
import datetime # Needed for one of the agent's internal time conversions

# Assuming agent.py, calendar_client.py, and ride_booking_client.py are in the same directory
# or accessible via PYTHONPATH.
from agent import VoiceAgent 

class TestVoiceAgent(unittest.TestCase):

    def setUp(self):
        """
        Set up an instance of VoiceAgent before each test method.
        """
        self.agent = VoiceAgent()

    @patch('agent.get_upcoming_events')
    def test_process_command_calendar_with_events(self, mock_get_events):
        """
        Test calendar command when there are events.
        """
        sample_event_time = datetime.datetime.now().isoformat()
        mock_get_events.return_value = [
            {'summary': 'Team Meeting', 'start_time': sample_event_time, 'end_time': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()}
        ]
        
        response = self.agent.process_command("What are my events for today?")
        self.assertIn("Team Meeting", response)
        self.assertIn(sample_event_time, response)
        mock_get_events.assert_called_once()

    @patch('agent.get_upcoming_events')
    def test_process_command_calendar_no_events(self, mock_get_events):
        """
        Test calendar command when there are no events.
        """
        mock_get_events.return_value = []
        
        response = self.agent.process_command("any events?")
        self.assertEqual("You have no upcoming events.", response)
        mock_get_events.assert_called_once()

    @patch('agent.book_ride')
    def test_process_command_ride_success(self, mock_book_ride):
        """
        Test ride booking command for a successful booking.
        """
        mock_book_ride.return_value = {
            'status': 'success', 
            'message': 'Ride to Quantum Realm at tomorrow 10 AM booked successfully.',
            'details': {'destination': 'Quantum Realm', 'time': 'tomorrow 10 AM'} # Mock details
        }
        
        # Test with "to" and "for"
        command = "book a ride to Quantum Realm for tomorrow 10 AM"
        response = self.agent.process_command(command)
        
        self.assertIn("Ride to Quantum Realm at tomorrow 10 AM booked successfully.", response)
        # The agent's _parse_destination_and_time will process "tomorrow 10 AM"
        # Depending on its exact logic, it might try to convert it or pass it as is.
        # For this test, we care that book_ride was called with the parsed values.
        mock_book_ride.assert_called_once_with("quantum realm", "tomorrow 10 am")


    @patch('agent.book_ride')
    def test_process_command_ride_success_with_at(self, mock_book_ride):
        """
        Test ride booking command for a successful booking using "at" for time.
        """
        mock_book_ride.return_value = {
            'status': 'success',
            'message': 'Ride to Mars Base at now booked successfully.',
            'details': {'destination': 'Mars Base', 'time': 'now'}
        }

        command = "Can you book a ride to Mars Base at now"
        response = self.agent.process_command(command)

        self.assertIn("Ride to Mars Base at now booked successfully.", response)
        mock_book_ride.assert_called_once_with("mars base", "now")


    @patch('agent.book_ride')
    def test_process_command_ride_success_destination_only(self, mock_book_ride):
        """
        Test ride booking command when only destination is provided (time defaults to ASAP).
        """
        # Mock datetime.now() to control the "ASAP" time
        fixed_now = datetime.datetime(2024, 7, 15, 10, 0, 0)
        
        with patch('agent.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = fixed_now
            iso_asap_time = fixed_now.isoformat()

            mock_book_ride.return_value = {
                'status': 'success',
                'message': f'Ride to The Cafe at {iso_asap_time} booked successfully.',
                'details': {'destination': 'The Cafe', 'time': iso_asap_time}
            }

            command = "I need a ride to The Cafe"
            response = self.agent.process_command(command)

            self.assertIn(f"Ride to The Cafe at {iso_asap_time} booked successfully.", response)
            mock_book_ride.assert_called_once_with("the cafe", iso_asap_time)


    @patch('agent.book_ride')
    def test_process_command_ride_failure_from_client(self, mock_book_ride):
        """
        Test ride booking when the ride_booking_client returns an error.
        """
        mock_book_ride.return_value = {
            'status': 'error', 
            'message': 'Sorry, no drivers available at this time.'
        }
        
        command = "book a ride to Atlantis for 3 AM"
        response = self.agent.process_command(command)
        
        self.assertIn("Sorry, no drivers available at this time.", response)
        mock_book_ride.assert_called_once_with("atlantis", "3 am")

    def test_process_command_unknown(self):
        """
        Test an unrecognized command.
        """
        response = self.agent.process_command("Tell me the weather forecast.")
        self.assertEqual("Sorry, I didn't understand that. I can help with calendar events and booking rides.", response)

    def test_process_command_empty(self):
        """
        Test an empty command string.
        """
        response = self.agent.process_command("")
        # Assuming empty command is also treated as "didn't understand"
        self.assertEqual("Sorry, I didn't understand that. I can help with calendar events and booking rides.", response)

if __name__ == '__main__':
    unittest.main()
