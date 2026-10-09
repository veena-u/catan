#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Resource {
    Brick,
    Lumber,
    Wool,
    Grain,
    Ore,
}

impl Resource {
    pub const ALL: [Resource; 5] =
        [Resource::Brick, Resource::Lumber, Resource::Wool, Resource::Grain, Resource::Ore];

    pub const fn index(self) -> usize {
        self as usize
    }
}

pub const RESOURCE_CARDS: u8 = 19;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_resource_is_at_its_index_in_increasing_order() {
        for (i, resource) in Resource::ALL.into_iter().enumerate() {
            assert_eq!(resource.index(), i);
        }
        assert!(Resource::ALL.windows(2).all(|w| w[0] < w[1]));
    }
}
