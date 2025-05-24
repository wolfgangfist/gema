import unittest
from ride_booking_client import book_ride # Assuming ride_booking_client.py is in the same directory or PYTHONPATH

class TestRideBookingClient(unittest.TestCase):

    def test_book_ride_success(self):
        destination = "Airport"
        time = "2024-07-16T10:00:00"
        response = book_ride(destination, time)
        
        self.assertEqual(response['status'], 'success')
        self.assertIn(destination, response['message'])
        self.assertIn(time, response['message'])
        self.assertIsInstance(response['details'], dict)
        self.assertEqual(response['details']['destination'], destination)
        self.assertEqual(response['details']['time'], time)
        self.assertIn('driver_name', response['details'])
        self.assertIn('license_plate', response['details'])

    def test_book_ride_failure_missing_destination(self):
        time = "2024-07-16T10:00:00"
        response = book_ride("", time)
        
        self.assertEqual(response['status'], 'error')
        self.assertIn("Destination and time are required", response['message'])

    def test_book_ride_failure_missing_time(self):
        destination = "Downtown"
        response = book_ride(destination, "")
        
        self.assertEqual(response['status'], 'error')
        self.assertIn("Destination and time are required", response['message'])

    def test_book_ride_failure_missing_both(self):
        response = book_ride("", "")
        
        self.assertEqual(response['status'], 'error')
        self.assertIn("Destination and time are required", response['message'])
        
    def test_book_ride_failure_none_time(self):
        destination = "Coffee Shop"
        response = book_ride(destination, None) # Test with None for time
        
        self.assertEqual(response['status'], 'error')
        self.assertIn("Destination and time are required", response['message'])

    def test_book_ride_failure_none_destination(self):
        time = "2024-07-16T11:00:00"
        response = book_ride(None, time) # Test with None for destination
        
        self.assertEqual(response['status'], 'error')
        self.assertIn("Destination and time are required", response['message'])

if __name__ == '__main__':
    unittest.main()
