db = db.getSiblingDB('open5gs');

// 기존 가입자 삭제 (있다면)
db.subscribers.deleteOne({imsi: "001010000000001"});

// 새 가입자 등록
db.subscribers.insertOne({
    "imsi": "001010000000001",
    "subscribed_rau_tau_timer": 12,
    "network_access_mode": 0,
    "subscriber_status": 0,
    "access_restriction_data": 32,
    "slice": [
        {
            "sst": 1,
            "sd": "000001",
            "default_indicator": true,
            "session": [
                {
                    "name": "internet",
                    "type": 3,  // IPv4
                    "qos": {
                        "index": 9,
                        "arp": {
                            "priority_level": 8,
                            "pre_emption_capability": 1,
                            "pre_emption_vulnerability": 1
                        }
                    },
                    "ambr": {
                        "downlink": {
                            "value": 1,
                            "unit": 3  // Gbps
                        },
                        "uplink": {
                            "value": 1,
                            "unit": 3  // Gbps
                        }
                    }
                }
            ]
        }
    ],
    "ambr": {
        "downlink": {
            "value": 1,
            "unit": 3  // Gbps
        },
        "uplink": {
            "value": 1,
            "unit": 3  // Gbps
        }
    },
    "security": {
        "k": "00112233445566778899AABBCCDDEEFF",
        "amf": "8000",
        "op": "E8ED289DEBA952E4283B54E88E6183CA",
        "opc": null
    },
    "__v": 0
});

print("Subscriber registered: IMSI 001010000000001");
