module HyNomasHistoryAttentionProbe
  # Release guards intentionally report different plan-specific prose. Match
  # the shared semantic signal while requiring the chain was still in flight.
  def self.inflight_chain_release_rejected?(chain_completed : Bool,
                                            error_message : String?) : Bool
    return false if chain_completed
    message = error_message || return false
    message.starts_with?("Complete ") && message.includes?(" chain")
  end
end
