//! The flag-set types: usages, format features, device features, sample counts and the like.
//!
//! Each is a `u32` newtype whose defined flags are associated constants. Unknown bits are refused
//! when a set is built from raw bits or deserialised ([`UnknownFlags`]), so a set that exists only
//! ever holds defined flags.
//!
//! [`UnknownFlags`]: crate::error::UnknownFlags

/// Defines a flag-set newtype over `u32`.
macro_rules! flag_set {
    (
        $(#[$meta:meta])*
        $name:ident {
            $( $(#[$flag_meta:meta])* $flag:ident = $bit:literal, )*
        }
    ) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
        )]
        #[serde(try_from = "u32", into = "u32")]
        pub struct $name(u32);

        impl $name {
            $( $(#[$flag_meta])* pub const $flag: Self = Self(1 << $bit); )*

            /// No flags.
            pub const EMPTY: Self = Self(0);

            /// Every defined flag.
            pub const ALL: Self = Self(0 $(| (1 << $bit))*);

            /// The raw bits.
            pub const fn bits(self) -> u32 {
                self.0
            }

            /// The set with these bits, or an error naming the bits that are no flag.
            pub const fn from_bits(bits: u32) -> Result<Self, $crate::error::UnknownFlags> {
                let unknown = bits & !Self::ALL.0;
                if unknown == 0 {
                    Ok(Self(bits))
                } else {
                    Err($crate::error::UnknownFlags { name: stringify!($name), bits: unknown })
                }
            }

            /// Whether every flag of `other` is set in `self`.
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Whether any flag of `other` is set in `self`.
            pub const fn intersects(self, other: Self) -> bool {
                self.0 & other.0 != 0
            }

            /// Both sets' flags.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// `self` without the flags of `other`.
            #[must_use]
            pub const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            /// Whether no flag is set.
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }
        }

        impl core::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl TryFrom<u32> for $name {
            type Error = $crate::error::UnknownFlags;
            fn try_from(bits: u32) -> Result<Self, Self::Error> {
                Self::from_bits(bits)
            }
        }

        impl From<$name> for u32 {
            fn from(set: $name) -> u32 {
                set.bits()
            }
        }
    };
}

pub(crate) use flag_set;

#[cfg(test)]
mod tests {
    use crate::caps::FormatFeatures;
    use crate::error::UnknownFlags;

    #[test]
    fn from_bits_refuses_unknown_bits() {
        let all = FormatFeatures::ALL.bits();
        assert_eq!(FormatFeatures::from_bits(all), Ok(FormatFeatures::ALL));
        let first_unknown = 1 << all.bit_width();
        assert_eq!(
            FormatFeatures::from_bits(all | first_unknown),
            Err(UnknownFlags {
                name: "FormatFeatures",
                bits: first_unknown
            })
        );
        assert_eq!(
            FormatFeatures::from_bits(first_unknown).map_err(|e| e.to_string()),
            Err(format!(
                "FormatFeatures has unknown bits {first_unknown:#x}"
            ))
        );
    }

    #[test]
    fn deserialising_goes_through_from_bits() {
        let all = FormatFeatures::ALL.bits();
        let bytes = postcard::to_allocvec(&all).expect("serialise");
        let back: FormatFeatures = postcard::from_bytes(&bytes).expect("deserialise");
        assert_eq!(back, FormatFeatures::ALL);

        let first_unknown = 1 << all.bit_width();
        let bytes = postcard::to_allocvec(&(all | first_unknown)).expect("serialise");
        assert!(postcard::from_bytes::<FormatFeatures>(&bytes).is_err());
    }

    #[test]
    fn set_operations() {
        let a = FormatFeatures::SAMPLE;
        let b = FormatFeatures::RENDER;
        let ab = a | b;
        assert!(ab.contains(a) && ab.contains(b) && !a.contains(ab));
        assert!(ab.intersects(a) && !a.intersects(b));
        assert_eq!(ab.difference(a), b);
        assert!(FormatFeatures::EMPTY.is_empty() && !a.is_empty());
        assert_eq!(u32::from(ab), ab.bits());
        assert_eq!(FormatFeatures::try_from(ab.bits()), Ok(ab));
    }
}
