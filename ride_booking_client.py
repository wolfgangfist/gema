import datetime

def book_ride(destination: str, time: str):
  """
  Simulates booking a ride to a given destination at a specified time.

  Args:
    destination: The destination address or location for the ride.
    time: The desired pickup time in ISO format (e.g., "2024-07-15T18:00:00").

  Returns:
    A dictionary containing the booking status and details.
    If successful, returns:
      {'status': 'success', 
       'message': f'Ride to {destination} at {time} booked successfully.', 
       'details': {'destination': destination, 
                   'time': time, 
                   'driver_name': 'Mock Driver', 
                   'license_plate': 'MOCK123'}}
    If destination or time is missing, returns:
      {'status': 'error', 
       'message': 'Destination and time are required for booking.'}
  """
  if not destination or not time:
    return {
        'status': 'error',
        'message': 'Destination and time are required for booking.'
    }

  # Simulate successful booking
  return {
      'status': 'success',
      'message': f'Ride to {destination} at {time} booked successfully.',
      'details': {
          'destination': destination,
          'time': time,
          'driver_name': 'Mock Driver',
          'license_plate': 'MOCK123'
      }
  }

if __name__ == "__main__":
  # Successful booking
  booking_time = (datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat()
  successful_booking = book_ride(destination="123 Main St, Anytown", time=booking_time)
  print("Successful Booking Attempt:")
  print(f"  Status: {successful_booking['status']}")
  print(f"  Message: {successful_booking['message']}")
  if successful_booking['status'] == 'success':
    print(f"  Details: {successful_booking['details']}")
  print("-" * 30)

  # Failed booking (missing destination)
  failed_booking_missing_dest = book_ride(destination="", time=booking_time)
  print("Failed Booking Attempt (Missing Destination):")
  print(f"  Status: {failed_booking_missing_dest['status']}")
  print(f"  Message: {failed_booking_missing_dest['message']}")
  print("-" * 30)

  # Failed booking (missing time)
  failed_booking_missing_time = book_ride(destination="456 Oak Ave, Anytown", time=None)
  print("Failed Booking Attempt (Missing Time):")
  print(f"  Status: {failed_booking_missing_time['status']}")
  print(f"  Message: {failed_booking_missing_time['message']}")
  print("-" * 30)

  # Failed booking (missing both)
  failed_booking_missing_both = book_ride(destination=None, time="")
  print("Failed Booking Attempt (Missing Both):")
  print(f"  Status: {failed_booking_missing_both['status']}")
  print(f"  Message: {failed_booking_missing_both['message']}")
  print("-" * 30)
